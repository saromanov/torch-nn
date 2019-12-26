def to_img(img, w, h):
    out = 0.5 * (img + 1)
    out = out.clamp(0, 1)
    out = out.view(-1, 1, w, h)
    return out
