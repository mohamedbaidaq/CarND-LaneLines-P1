# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

#### 1. Describtion of the pipeline. 
##### My pipeline consisted of 6 steps: 

###### 1) convert the image to gray scale
gray = grayscale(image)
###### 2) Define a kernel size and apply Gaussian smoothing
kernel_size = 5
blur_gray = gaussian_blur(gray, kernel_size)
###### 3) Define our parameters for Canny and apply
low_threshold = 50
high_threshold = 150
edges = canny(blur_gray, low_threshold, high_threshold)
###### 4) Next we'll create a masked edges image using region_of_interest() helper function
###### defining a four sided polygon to mask
imshape = image.shape
vertices = np.array([[(0,imshape[0]),(450, 325), (500, 325), (imshape[1],imshape[0])]], dtype=np.int32)    
masked_edges = region_of_interest(edges, vertices)
###### 5) Define the Hough transform parameters
###### Make a blank the same size as our image to draw on
rho = 2 # distance resolution in pixels of the Hough grid
theta = np.pi/180 # angular resolution in radians of the Hough grid
threshold = 15     # minimum number of votes (intersections in Hough grid cell)
min_line_length = 10 # minimum number of pixels making up a line
max_line_gap = 20    # maximum gap in pixels between connectable line segments
###### Run Hough using the helper function hough_lines() on edge detected image
###### Output "lines" is an array containing endpoints of detected line segments
line_image = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
###### 6) Draw the lines on the "Original" image using the helper function weighted_img()
lines_orig_RGB = weighted_img(line_image, image)
##### In order to draw a single line on the left and right lanes, I modified the draw_lines() as follows: 
##### So Instead of just drawing every detected line by Hough transform, I've done the following steps: 
###### 1) Loop over the detected lines/segments from the Hough transform, and do the following
    1.1) Detect whether the segment/line is belonging to the left lane or the right lane by detcting the slope of the segment
        If the slope is negative, then the segment belongs to the left lane
        If the slope is positive, then the segments belongs to the right lan
    1.2) Get the length of the detected segment, and save the coordinates of longest segment on every lane
###### 2) Derive the longest segment equation by using np.polyfit or by using Y = mx + b line equation. 
###### 3) Get the intersection between Region of Interest borders, and the two identified longest segments on every lane.
###### 4) Use the new intersctions as Coordinates to draw the final solid lane 

###### find the longest segment in every lane, and derive the slope out of it
lsll = [0,0,0,0] # longest segment on the left lane
lsrl = [0,0,0,0] # longest segment on the right lane
for line in lines:
    for x1,y1,x2,y2 in line:
        if ((((y2-y1)/(x2-x1)) < 0) and ((x2-x1) != 0)): # then it's a negative slope, Thus it's  left lane segment
            cs_len = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            lsll_len = math.sqrt((lsll[2] - lsll[0])**2 + (lsll[3] - lsll[1])**2)
            if(cs_len > lsll_len):
                lsll = [x1, y1, x2, y2]
        elif ((((y2-y1)/(x2-x1)) > 0) and ((x2-x1) != 0)): # otherwise, it's a positive slope, i.e. its's right lane segment
            cs_len = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            lsrl_len = math.sqrt((lsrl[2] - lsrl[0])**2 + (lsrl[3] - lsrl[1])**2)
            if(cs_len > lsrl_len):
                lsrl = [x1, y1, x2, y2]
###### now derive the segment line equation
fit_left_lane = np.polyfit((lsll[0], lsll[2]), (lsll[1], lsll[3]), 1) # left lane
fit_right_lane = np.polyfit((lsrl[0], lsrl[2]), (lsrl[1], lsrl[3]), 1) # right lane
###### now find the intersections
top_left_pt_X = int((325 - fit_left_lane[1])/fit_left_lane[0])
bottom_left_pt_X = int((img.shape[0] - fit_left_lane[1])/fit_left_lane[0])
top_right_pt_X = int((325 - fit_right_lane[1])/fit_right_lane[0])
bottom_right_pt_X = int((img.shape[0] - fit_right_lane[1])/fit_right_lane[0])
###### now draw the full lanes based on the new intersections 
cv2.line(img, (top_left_pt_X, 325),(bottom_left_pt_X, img.shape[0]), color, thickness=5) # left lane 
cv2.line(img, (top_right_pt_X, 325),(bottom_right_pt_X, img.shape[0]), color, thickness=5) # right lane

##### In order to compare the original image before processing, and the final image after processing the lane side to side,
##### I've implemented the following helper function 

def display_images(input_dir, output_dir): 
    for img_idx, output_image in enumerate(os.listdir(input_dir)):
        f = plt.figure()
        f.add_subplot(1,2, 1)
        input_image = os.listdir(input_dir)[img_idx]
        image = mpimg.imread(input_dir+input_image)
        plt.imshow(image)

        f.add_subplot(1,2, 2)
        output_image = os.listdir(output_dir)[img_idx]
        image = mpimg.imread(output_dir+output_image)
        plt.imshow(image)

##### In order to read the images from the input folder, and write the output in another output folder, I've implemented the following routine

def find_lanes(input_dir, output_dir):
    for image_name in os.listdir(input_dir): 
        orig_image = mpimg.imread(input_dir+image_name)
        processed_image = Lane_Finding_Pipeline(orig_image)
        # and now convert the image to RGB format         
        output_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # write the resulting image first into the output directory 
        cv2.imwrite(output_dir+image_name, output_image)


### 2. Identify potential shortcomings with your current pipeline

1) the pieline is not perfectly working when there are curves on the road like in the challenge video. 
2) manipulating the parameters for Hough Transform is tricky, and it needs some practice, experience, any maybe another approach like 
implementing some parameter tuner. 

### 3. Suggest possible improvements to your pipeline

A possible improvement would be to implement a parameter tuner in order to get the perfect parameters for the canny/ hough transform. 
Another potential improvement could be to use deep learning to train a neural network that can identify the lanes. 
