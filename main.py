import cv2
import os

sample = cv2.imread("SOCOFing/Altered/Altered-Hard/150__M_Left_middle_finger_Obl.BMP")

best_score = 0
filename = None
image = None

kp1, kp2, mp = None, None, None

# Create SIFT object
sift = cv2.SIFT_create()

# Compute keypoints and descriptors for the sample image
keypoints_1, descriptors_1 = sift.detectAndCompute(sample, None)

# Iterate through the first 200 files in the "Real" directory
for file in os.listdir("SOCOFing/Real")[:2000]:
    fingerprint_image = cv2.imread(os.path.join("SOCOFing/Real", file))
    
    # Compute keypoints and descriptors for the fingerprint image
    keypoints_2, descriptors_2 = sift.detectAndCompute(fingerprint_image, None)
    
    # Use FLANN-based matcher to find matches
    index_params = dict(algorithm=1, trees=10)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)
    
    # Apply ratio test to find good matches
    match_points = []
    for p, q in matches:
        if p.distance < 0.75 * q.distance:  # Adjust the ratio as needed
            match_points.append(p)
    
    # Calculate the ratio of good matches to keypoints
    keypoints = min(len(keypoints_1), len(keypoints_2))
    if keypoints == 0:  # Avoid division by zero
        continue
    
    score = len(match_points) / keypoints * 100
    
    # Update the best match if the current score is higher
    if score > best_score:
        best_score = score
        filename = file
        image = fingerprint_image
        kp1, kp2, mp = keypoints_1, keypoints_2, match_points

print("BEST MATCH: " + filename)
print("SCORE: " + str(best_score))

# Draw the best matches
if kp1 and kp2 and mp:
    result = cv2.drawMatches(sample, kp1, image, kp2, mp, None)
    result = cv2.resize(result, None, fx=4, fy=4)
    cv2.imshow("result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No matches found.")
