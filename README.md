
# Microscopy deblurring 

## Synopsis: 
Capturing a full scan of an eye, for medical examination, is very
challenging. As visualized in Fig. 2, the cornia has a spherical shape, but an
imaging system can only focus (at its maximum sharpness) on a given depth
range in the scene. You might have witnessed this when capturing photos, all
objects closer or further away from the object at which you focus your camera
end up blurry/out of focus. While this can be used for artistic effects in photography, it is very detrimental in biomedical imaging. The goal of this project is to
deblur the cornia scans, as much as possible while reconstructing faithfully the
original signal. It is probably not possible to deblur the entire range of depth
values, but the more we can extend the depth of field of the imaging system,
the fewer shots need to be captured, and the less time patients need to sit in
front of the imaging system. One challenge in this project is that you are not
provided with a lot of data. Your approach should take that into account, and
could possibly be developed as a method that needs retraining for new data
(this can also be a good advantage of your method, since it means it can apply
well to new imaging systems, or imaged content, without the need for a large
dataset each time).
