import numpy as np 
from scipy.spatial.distance import cdist
import skimage
from scipy.ndimage import gaussian_filter

class PFCM:
    """
        Fuzzy possibility C means 
    ----------------------------------------------------
        n_clusters : number of clusters 
        max_iter : max number of iterations
        tol : termination tolerance / error threshold 
        membership : fuzziness for membership
        eta : fuzziness for typicalities 
        a, b : weightings for membership vs typicalities
        init : initialization for u 
        random_state : for reproducibility 

    """
    def __init__(
        self,
        n_clusters : int = 3,
        max_iter : int = 50,
        tol : float = 0.005, 
        membership : int = 2, 
        eta : int = 2 ,
        a : int = 1,
        b : int = 1,
        init : str = "FCM", 
        random_state = None) -> None:

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol 
        self.membership = membership 
        self.eta = eta 
        self.a = a 
        self.b = b 
        self.init = init 
        self.rng = np.random.default_rng(random_state)

    def _initialize_u(self, X):
        """
            initialization for fuzzy memberships 
        """
        # num of data points and num of clusters 
        n = X.shape[0]
        c = self.n_clusters

        # random initialization
        if self.init == "random":
            u = self.rng.random((n, c))
            u /= u.sum(axis=1, keepdims=True)
        
        # FCM initialization 
        else:
            u = self.rng.random((n, c))
            u /= u.sum(axis=1, keepdims=True)
            
            # 5 loops of FCM
            for _ in range(5):
                
                # compute cluster centres v( C,d  ) 
                # Equation 4b of (N.pal et al 2005)
                v = (u.T ** self.membership) @ X / np.sum(u ** self.membership, axis = 0, keepdims = True).T

                # compute the euclidean distance 
                dist = cdist(X, v, "euclidean")

                # prevent zero 
                dist = np.maximum(dist, 1e-20)

                # update fuzzy membership
                # standard FCM update 
                u = 1.0 / ( dist ** (2/(self.membership - 1))) / np.sum(1.0 / dist ** (2/(self.membership - 1)), axis = 1, keepdims= True)

        return u 
            
    def fit(self, X):
        """
            fit PFCM to data 
        """
        X = np.atleast_2d(X)

        n, _ = X.shape
        c = self.n_clusters

        u = self._initialize_u(X)
        
        # initial typicalities
        t = np.ones((n, c)) * 0.5

        for it in range(self.max_iter):
            u_old = u.copy()

            # assign weighting for membership vs typicalities
            a_weight = self.a * (u ** self.membership)
            b_weight = self.b * (t ** self.eta)

            weights = a_weight + b_weight

            # update u with typicalities weightings
            # v equation from equation 23, N. pal et al 2005
            v = (weights.T @ X) / weights.sum(axis = 0, keepdims = True).T
            
            # calculate euclidean distance
            dist = cdist(X, v, "euclidean")
            
            # from eq 11
            gamma = (weights * (dist ** 2)).sum(axis = 0) / weights.sum(axis = 0)
            gamma = np.maximum(gamma, 1e-20)
            self.gamma = gamma

            # probabilistic membership 
            # if points are close compared to gamma t -> 1, if points are far t -> 0 
            exponent = 1.0 / (self.eta - 1)
            t = 1.0 / ( 1.0 + ( self.b * (dist**2) / gamma) ** exponent)

            # update u, this is the same as the standard FCM update  
            dist_pow = dist ** ( 2 / (self.membership - 1))
            u = 1.0 / ( dist_pow * (1.0 /dist_pow).sum(axis=1, keepdims= True))

            # convergence check, stops if it converges
            if np.max(np.abs(u - u_old)) < self.tol:
                break 

        self.v_ = v
        self.u_ = u 
        self.t_ = t 
        self.n_iter = it + 1 

        return self

    def predict(self, X):
        """ 
            get values for new data points
        """
        X = np.atleast_2d(X)

        dist = cdist(X, self.v_, "euclidean")

        u = 1.0 / ( dist ** (2/(self.membership - 1))) / np.sum(1.0 / dist ** (2/(self.membership - 1)), axis = 1, keepdims= True)

        gamma = self.gamma
        
        exponent = 1.0 / (self.eta - 1)

        t = 1.0 / ( 1.0 + ( self.b * (dist**2) / gamma) ** exponent)

        return u, t 


def pfcm_intensity_norm(img, n_clusters, target_ints = None, sample_size = 10000, **pfcm_kwargs):

    img = skimage.util.img_as_float(img)

    h, w = img.shape

    pixels = img.ravel().reshape(-1, 1)

    # speed up sampling 
    if sample_size is not None and pixels.shape[0] > sample_size:
        idx = np.random.default_rng(0).choice(pixels.shape[0], sample_size, replace = False)
        X_sample = pixels[idx]
    else:
        X_sample = pixels

    # predict clusters 
    pfcm = PFCM(n_clusters = n_clusters, **pfcm_kwargs)
    pfcm.fit(X_sample)
    centres = pfcm.v_.ravel()

    # sort by cluster intensity
    order = np.argsort(centres)
    centres = centres[order]
    u = pfcm.u_[:, order] # original membership matrix reordered

    # define target intensity if not given 

    if target_ints is None:
        target_ints = np.linspace(0, 1, n_clusters)
    
    # normalisation: weighted sum of target intensities 
    # for full image compute pfcm with predict
    if sample_size is not None and sample_size < pixels.shape[0]:
        # recompute membership for all pixels 

        dist = cdist(pixels, pfcm.v_[order], 'euclidean')
        u_full = 1.0 / (dist ** ( 2 / (pfcm.membership -1))) / np.sum(1.0 / (dist ** (2 / (pfcm.membership -1))), axis = 1, keepdims= True)

    else:
        u_full = u 

    normalized_flat = u_full @ target_ints
    normalized = normalized_flat.reshape(h, w)

    u_map = u_full.reshape(h, w, n_clusters)

    return normalized, centres, u_map 

class BCPFCM(PFCM):
    """
        Bias corrected PFCM
        Adds a spatially multiplictive bias field B(x,y)

        bias_smooth_sigma: controls how smooth the estimate bias field is. 
        Larger value means the bias field can only vary slowly (gentle vignetting). Smaller value indicates sharper changes 

        Observed image is observed as 
        I(x,y) = J(x,y) * B(x,y)
        I is the observed image, 
        J is the true image,
        B is the mulitplictive field ( bias ) that is applied to the true image to obtain the observed image

    """
    def __init__(self,
        n_clusters : int = 3,
        max_iter : int = 50,
        tol : float = 0.005, 
        membership : int = 2, 
        eta : int = 2 ,
        a : int = 1,
        b : int = 1,
        init : str = "FCM",
        bias_smooth_sigma = 8,
        random_state = None) -> None:
        super().__init__(n_clusters, max_iter, tol, membership, eta, a, b, init, random_state)
        self.bias_smooth_sigma = bias_smooth_sigma

    @staticmethod
    def _check_shape(X_flat, img_shape):
        h, w = img_shape
        if h * w != len(X_flat):
            raise ValueError(
                f"img_shape {img_shape} implies {h * w} pixels but X has {len(X_flat)}."
            )

    # refactor helper functions 
    def _smooth_bias(self, B_raw, img_shape):
        """Smooth a flat bias estimate and normalise by its median."""
        h, w = img_shape
        B = gaussian_filter(B_raw.reshape(h, w), sigma=self.bias_smooth_sigma)
        # Median normalisation: robust to large background / air regions
        nonzero = B[B > 0]
        median = np.median(nonzero) if nonzero.size > 0 else 1.0
        B -= median + 1e-20
        return B                                    # (H, W)
 
    def _membership_and_typicalities(self, I_corr_1d, v, gamma):
        """Return (u, t) given bias-corrected 1-D intensities, centres, and gamma."""
        dist = np.maximum(
            cdist(I_corr_1d[:, np.newaxis], v, "euclidean"), 1e-20
        )
        dist_pow = dist ** (2.0 / (self.membership - 1))
        u = 1.0 / (dist_pow * (1.0 / dist_pow).sum(axis=1, keepdims=True))
 
        exponent = 1.0 / (self.eta - 1)
        t = 1.0 / (1.0 + (self.b * dist ** 2 / gamma) ** exponent)
 
        return u, t
 
    def _estimate_gamma(self, u, t, dist):
        """Gamma using the full combined weight a·u^m + b·t^η (Eq. 11)."""
        a_weight = self.a * (u ** self.membership)
        b_weight = self.b * (t ** self.eta)
        weights = a_weight + b_weight
        gamma = (weights * dist ** 2).sum(axis=0) / weights.sum(axis=0)
        return np.maximum(gamma, 1e-20)
 
    def _update_bias(self, X_raw_1d, u, v, img_shape):
        """Estimate, smooth, and return the bias field as an (H, W) array."""

        log_x = np.log(X_raw_1d + 1e-20)
        log_pred = np.log(((u*v[:, 0]).sum(axis=1)) + 1e-20)
        log_b_raw = log_x - log_pred 
        log_b = self._smooth_bias(log_b_raw, img_shape)
        
        b = np.exp(log_b)

        return b 

        # pred_intensity = (u * v[:, 0]).sum(axis=1)   # (n,)
        # B_raw = X_raw_1d / (pred_intensity + 1e-20)
        # return self._smooth_bias(B_raw, img_shape)    # (H, W)
 
    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
 
    def fit(self, X, img_shape, sample_size=None):
        """
        Fit the bias-corrected PFCM model.
 
        Parameters
        ----------
        X           : 1-D array-like of raw pixel intensities (n_pixels,).
        img_shape   : (H, W) – used for spatial smoothing of the bias field.
        sample_size : int or None.
                      • None  → full iterative bias correction (accurate).
                      • int   → fit centres on a subsample then apply two
                        single-pass bias-correction steps (faster, less
                        accurate for strong fields).
        """
        X = np.asarray(X, dtype=float).ravel()      # always 1-D from here on
        self._check_shape(X, img_shape)
 
        if sample_size is not None and X.size > sample_size:
            self._fit_sampled(X, img_shape, sample_size)
        else:
            self._fit_full(X, img_shape)
 
        return self
 
    def _fit_full(self, X, img_shape):
        """Full iterative bias correction on all pixels.
 
        Parameters
        ----------
        X         : 1-D float array (n_pixels,) – raw intensities.
        img_shape : (H, W)
        """
        h, w = img_shape
        n = X.size
        c = self.n_clusters
 
        bias = np.ones(n)                            # initial bias field (flat)
        u = self._initialize_u(X[:, np.newaxis])     # needs (n, 1)
        t = np.ones((n, c)) * 0.5
        u_old = u.copy()
 
        for it in range(self.max_iter):
            # 1. Bias-corrected intensities
            I_corr = X / (bias + 1e-20)              # (n,)
 
            # 2. Cluster centres in corrected space
            a_weight = self.a * (u ** self.membership)
            b_weight = self.b * (t ** self.eta)
            weights = a_weight + b_weight            # (n, C)
            v = (
                (weights.T @ I_corr[:, np.newaxis])          # (C, 1)
                / weights.sum(axis=0)[:, np.newaxis]         # (C, 1)
            )
 
            # 3. Distances in corrected space
            dist = np.maximum(
                cdist(I_corr[:, np.newaxis], v, "euclidean"), 1e-20
            )
 
            # 4. Gamma, typicalities, memberships
            gamma = self._estimate_gamma(u, t, dist)
            self.gamma = gamma
 
            exponent = 1.0 / (self.eta - 1)
            t = 1.0 / (1.0 + (self.b * dist ** 2 / gamma) ** exponent)
 
            dist_pow = dist ** (2.0 / (self.membership - 1))
            u = 1.0 / (dist_pow * (1.0 / dist_pow).sum(axis=1, keepdims=True))
 
            # 5. Update bias field from raw X and current u / v
            B_map = self._update_bias(X, u, v, img_shape)  # (H, W)
            bias = B_map.ravel()
 
            # 6. Convergence
            if it > 0 and np.max(np.abs(u - u_old)) < self.tol:
                break
            u_old = u.copy()
 
        self.v_ = v
        self.u_ = u
        self.t_ = t
        self.gamma = gamma
        self.bias_field_ = B_map                    # (H, W)
        self.n_iter_ = it + 1
 
    def _fit_sampled(self, X, img_shape, sample_size):
        """
        Fast two-phase fitting.
 
        Phase 1 - plain PFCM on a random subsample → initial centres.
        Phase 2 - two single-pass bias-correction steps over all pixels,
                  with centre refinement in between.
 
        Parameters
        ----------
        X           : 1-D float array (n_pixels,) – raw intensities.
        img_shape   : (H, W)
        sample_size : int
        """
        h, w = img_shape
        n = X.size
 
        # Use self.rng (seeded in __init__ via random_state) for reproducibility
        idx = self.rng.choice(n, sample_size, replace=False)
 
        # --- Phase 1: plain PFCM on subsample ----------------------------
        temp = PFCM(
            n_clusters=self.n_clusters, max_iter=self.max_iter,
            tol=self.tol, membership=self.membership, eta=self.eta,
            a=self.a, b=self.b, init=self.init,
        )
        temp.fit(X[idx, np.newaxis])
        v = temp.v_                                  # (C, 1)
        gamma = temp.gamma                           # (C,)
 
        # --- Pass 1: memberships from raw X (bias ≈ 1) → first B estimate
        u, t = self._membership_and_typicalities(X, v, gamma)
        B_map = self._update_bias(X, u, v, img_shape)   # (H, W)
        I_corr = X / (B_map.ravel() + 1e-20)
 
        # Refine centres in corrected space
        a_weight = self.a * (u ** self.membership)
        b_weight = self.b * (t ** self.eta)
        weights = a_weight + b_weight
        v = (
            (weights.T @ I_corr[:, np.newaxis])
            / weights.sum(axis=0)[:, np.newaxis]
        )
 
        # Recompute gamma with refined centres
        dist = np.maximum(
            cdist(I_corr[:, np.newaxis], v, "euclidean"), 1e-20
        )
        gamma = self._estimate_gamma(u, t, dist)
 
        # --- Pass 2: final u, t and bias field with refined centres ------
        u, t = self._membership_and_typicalities(I_corr, v, gamma)
        B_map = self._update_bias(X, u, v, img_shape)   # (H, W)
 
        self.v_ = v
        self.u_ = u
        self.t_ = t
        self.gamma = gamma
        self.bias_field_ = B_map                    # (H, W)
        self.n_iter_ = 1



    def correct(self, X, img_shape):
        """
            return bias corrected image J = I / B
        """
        X_flat = np.asarray(X, dtype=float).ravel()
        self._check_shape(X_flat, img_shape)
        return (X_flat / (self.bias_field_.ravel() + 1e-20)).reshape(img_shape)


def bcpfcm_norm(img, n_clusters, smooth_sigma = 8, sample_size = None, **kwargs):
    img = skimage.util.img_as_float(img)
    h,w = img.shape
    pixels = img.ravel()

    # fit bias correction PFCM

    model = BCPFCM(n_clusters = n_clusters, bias_smooth_sigma = smooth_sigma,  **kwargs)
    model.fit(pixels, img_shape = (h,w), sample_size = sample_size)

    # bias corrected image 
    corrected_image = model.correct(pixels, img_shape = (h,w))

    return corrected_image, model.bias_field_



    
