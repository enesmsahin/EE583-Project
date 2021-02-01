import Augmentor
p = Augmentor.Pipeline("./Dataset/Validation/Metu_red", save_format="jpg")
p.rotate(probability=0.5, max_left_rotation=10, max_right_rotation=10)
p.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)
p.shear(probability=0.5, max_shear_left=15, max_shear_right=15)
p.flip_left_right(probability=0.5)
p.random_brightness(probability=0.5, min_factor=0.5, max_factor=1.5)

p.sample(100)
