import keras.backend as K

def L1(y_true, y_pred):
    return K.mean(K.abs(y_true - y_pred))

def L2(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))

def PSNRLoss(y_true, y_pred):
    y_true = (y_true + 1.0) * 127.5
    y_pred = (y_pred + 1.0) * 127.5
    # y_true and y_pred range from -1 to 1
    return 10 * K.log((255**2)/(K.mean(K.square(y_pred - y_true))))

def SSIM_backup(y_true, y_pred):
    u_true = K.mean(y_true)
    u_pred = K.mean(y_pred)
    var_true = K.var(y_true)
    var_pred = K.var(y_pred)
    std_true = K.sqrt(var_true)
    std_pred = K.sqrt(var_pred)
    c1 = K.square(0.01*7)
    c2 = K.square(0.03*7)
    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
    return ssim / denom

def SSIM(y_true, y_pred):
    y_true = (y_true + 1.0) * 127.5
    y_pred = (y_pred + 1.0) * 127.5
    mean_true = K.mean(y_true)
    mean_pred = K.mean(y_pred)
    var_true = K.var(y_true)
    var_pred = K.var(y_pred)
    covar = K.mean((y_true - mean_true) * (y_pred - mean_pred))
    
    c1 = 0.01**2
    c2 = 0.03**2
    
    ssim = (2 * mean_true * mean_pred + c1) * (2 * covar + c2)
    denom = (mean_true ** 2 + mean_pred ** 2 + c1) * (var_pred + var_true + c2)
    return ssim / denom

def _Eval(t):
    return t.eval(session=K.get_session())

if __name__=='__main__':
    import numpy as np
    img1 = np.random.random(size=(5, 24, 24, 3))
    img2 = img1.copy()
    
    image1 = K.constant(img1)
    image2 = K.constant(img2)
    r = SSIM(image1, image2)
    print(_Eval(r))
    
    