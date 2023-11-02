import tensorflow as tf

def gradient(dy, dx, grad_ys=None):
    if grad_ys is None:
        dy_dx = tf.gradients(dy, dx)
    else:
        dy_dx = tf.gradients(dy, dx, grad_ys=grad_ys)
    if len(dy_dx)==1:
        dy_dx = dy_dx[0]
    return dy_dx

def fwd_gradient(dy, dx):
    dummy = tf.ones_like(dy)
    G = tf.gradients(dy, dx, grad_ys=dummy)[0]
    Y_x = tf.gradients(G, dummy)[0]
    return Y_x