# Virtual Makeup
OpenCV Course 2 (Project 1): Virtual Makeup

## Feature 1: Apply Lipstick

To use openCV and dlib to apply lipstick to a static image, I created a mask of the lips in the image by identifying the landmarks of the mouths. I then made a blur of the mask to obtain a more natural result so there isn't a sharp constrast between the lipstick and the face. I then applied the color mapping onto the masks. 

```python
def lipstick_mask(im,lipstick_r,lipstick_g,lipstick_b):

  #identify lip landmarkers
  lipsPoints = points[48:60]

  #create a lipstick mask
  mask = np.zeros((im.shape[0], im.shape[1], 3), dtype=np.float32)
  cv2.fillConvexPoly(mask, np.int32(lipsPoints), (lipstick_r,lipstick_g,lipstick_b))

  # Apply close operation to improve mask
  mask = 255*np.uint8(mask)
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,40))

  # Blur the mask to obtain natural result
  mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 1)
  # Calculate inverse mask
  mask = cv2.GaussianBlur(mask,(15,15),cv2.BORDER_DEFAULT)

  # Convert masks to float to perform blending
  inverseMask = cv2.bitwise_not(mask)
  mask = mask.astype(float)/255

  # Apply color mapping for the lips
  inverseMask = inverseMask.astype(float)/255

  # Convert lips and face to 0-1 range
  lips = cv2.applyColorMap(im, cv2.COLORMAP_INFERNO)
  lips = lips.astype(float)/255

  # Multiply lips and face by the masks
  ladyFace = im.astype(float)/255
  justLips = cv2.multiply(mask, lips)

  # Add face and lips
  justFace = cv2.multiply(inverseMask, ladyFace)
  result = justFace + justLips
  return result
 
 ```

## Feature 2: Apply Eye-Liner

In order to use openCV and dlib to apply eyeliner to the eyeline in an image, I first identified the landmarks of the eyes. I used "shape_predictor_68_face_landmarks.dat" to identified the facial landmarks (getEyeLandmarkPoints). The dlib 68-face-landmark vector only provides the landmarks of the top, bottom and corners of the eyes, so I created a function that interpolates the landmarks in between (getEyelinerPoints). Once the interpolated landmarkers were identified, I used cv2.line to draw the eyeliner on the image (addEyeliner), using the cv2.line I could choose the color and thickness of the eyeliner. 

```python 

def getEyeLandmarkPoints(face_landmark_points):
    #first we get all of the facial landmark features
    #then we will only select the eye landmarks
    face_landmark_points[36][0]-=5
    face_landmark_points[39][0]+=5
    face_landmark_points[42][0]-=5
    face_landmark_points[45][0]+=5
    
    L_eye_top = face_landmark_points[36: 40]
    L_eye_bottom = np.append(face_landmark_points[39: 42], face_landmark_points[36]).reshape(4,2)

    R_eye_top = face_landmark_points[42:  46]
    R_eye_bottom = np.append(face_landmark_points[45:48], face_landmark_points[42]).reshape(4,2)
       
    return [L_eye_top, L_eye_bottom, R_eye_top, R_eye_bottom]

def getEyelinerPoints(eye_landmark_points):
    #After identifying the eye landmarks, we will extrapolate the coordinates between the landmarks
    #This will be used to identify the eyeliner points
    L_eye_top, L_eye_bottom, R_eye_top, R_eye_bottom = eye_landmark_points

    L_interp_x = np.arange(L_eye_top[0][0], L_eye_top[-1][0], 1)
    R_interp_x = np.arange(R_eye_top[0][0], R_eye_top[-1][0], 1)

    L_interp_top_y = interpolateCoordinates(L_eye_top, L_interp_x)
    L_interp_bottom_y = interpolateCoordinates(L_eye_bottom, L_interp_x)

    R_interp_top_y = interpolateCoordinates(R_eye_top, R_interp_x)
    R_interp_bottom_y = interpolateCoordinates(R_eye_bottom, R_interp_x)

    return [(L_interp_x, L_interp_top_y, L_interp_bottom_y), (R_interp_x, R_interp_top_y, R_interp_bottom_y)]

def addEyeliner(img, color,thickness,interp_pts):
    #Here we choose the color and thickness of the eyeliner.
    L_eye_interp, R_eye_interp = interp_pts
    L_interp_x, L_interp_top_y, L_interp_bottom_y = L_eye_interp
    R_interp_x, R_interp_top_y, R_interp_bottom_y = R_eye_interp


    for i in range(len(L_interp_x)-2):
        x1 = L_interp_x[i]
        y1_top = L_interp_top_y[i]
        x2 = L_interp_x[i+1]
        y2_top = L_interp_top_y[i+1]
        cv2.line(img, (x1, y1_top), (x2, y2_top), color, thickness)

        y1_bottom = L_interp_bottom_y[i]
        y2_bottom = L_interp_bottom_y[i+1]
        cv2.line(img, (x1, y1_bottom), (x1, y2_bottom), color, thickness)

    
    for i in range(len(R_interp_x)-2):
        x1 = R_interp_x[i]
        y1_top = R_interp_top_y[i]
        x2 = R_interp_x[i+1]
        y2_top = R_interp_top_y[i+1]
        cv2.line(img, (x1, y1_top), (x2, y2_top), color, thickness)

        y1_bottom = R_interp_bottom_y[i]
        y2_bottom = R_interp_bottom_y[i+1]
        cv2.line(img, (x1, y1_bottom), (x1, y2_bottom), color, thickness)

    return img
    
    
```
