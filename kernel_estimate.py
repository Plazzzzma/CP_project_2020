def blind_kernel_estimate(Y, X_l, kernel_width, reg_mode=0, reg_weight=1):
    '''
    Operation: estimate the kernel k that minimizes ||Y-X_l*k||**2 (+ reg_weight * ||k||**2)
    Inputs: 
        2D images Y and X_l (Gray or multichannel)
        kernel_width (integer > 0, better if even)
        reg_mode (0: no reg, 1: L2 reg)
        reg_weight (weight of the L2 reg term, ignored when reg_mode=0)
    Outputs: 
        k of size kernel_width x kernel_width (or kernel_width-1 if it is odd)
    '''
    
    # Convert inputs to Fourier domain
    X_l_Freq = np.fft.fft2(X_l, axes=[0, 1])
    Y_Freq = np.fft.fft2(Y, axes=[0, 1])

    # Solve for k in Fourier domain (regularization only affects den)
    num = X_l_Freq.conjugate() * Y_Freq
    if reg_mode == 0:
        den = np.abs(X_l_Freq)**2 # Fourier transform of X_l transpose * X_l
    elif reg_mode == 1:
#         reg_term = reg_weight * np.identity(kernel_width)
#         reg_term_Freq = psf2otf(reg_term, Y.shape[:2])
#         if X_l_Freq.ndim == 3:
#             reg_term_Freq = np.repeat(reg_term_Freq[:, :, np.newaxis], X_l_Freq.shape[2], axis=2)
#         den = reg_term_Freq + np.abs(X_l_Freq)**2 # Fourier transform of [2*reg_weight + X_l transpose * X_l]
        den = reg_weight + np.abs(X_l_Freq)**2 # Fourier transform of [2*reg_weight + X_l transpose * X_l]
    k_l_Freq = num / den

    # Get average channel solution if multi-channel
    if k_l_Freq.ndim == 3:
        k_l_Freq = np.mean(k_l_Freq, 2)
    
    # Convert back to spatial, given the width
    if kernel_width < 1:
        raise ValueError('kernel_width must be a positive integer')
    k_l = otf2psf(k_l_Freq, [kernel_width, kernel_width])

    # Correct the pixel shift for odd width
    if (kernel_width % 2 == 1):
        k_l = k_l[1:,1:]
    
    # Normalize to 1
    k_l = k_l / k_l.sum()
    
    return k_l
