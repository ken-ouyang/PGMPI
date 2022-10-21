import subprocess
from pathlib import Path
import glob
from PIL import Image
from PIL import ImageFilter
import numpy as np
import mediapipe as mp
import cv2
from scipy.stats import multivariate_normal

data_dir ="sample_scene"
videos_path = glob.glob("%s/*mp4"%(data_dir))
image_path_o = f"{data_dir}/images"
ref_video = 'video_0'
fps = 3 

subprocess.run(["mkdir", "-p", str(image_path_o)])

max_scale = 1.0  # @param {type:'number'}
# filters = f"mpdecimate,setpts=N/FRAME_RATE/TB,scale=iw*{max_scale}:ih*{max_scale}"

for video_path in videos_path:
    video_name = video_path.split("/")[-1].split(".")[0] 
    out_pattern = image_path_o +  f'/%06d_{video_name}.png'
    # subprocess.run(["ffmpeg", "-i", video_path, "-r", str(fps), "-vf", filters, out_pattern])
    subprocess.run(["ffmpeg", "-i", video_path, "-r", str(fps),  out_pattern])

# get mask
import torch
from torchvision import transforms
model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
model.eval()

mask_path = f"{data_dir}/masks"
Path(mask_path).mkdir(exist_ok=True, parents=True)
file_list = glob.glob("%s/*"%image_path_o)
for filename in file_list:
    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    # run models 
    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)

    # create a color pallette, selecting a color for each class
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    # plot the semantic segmentation predictions of 21 classes in each color
    out = output_predictions.byte().cpu().numpy()
    mask_h = (out==15)
    mask_o = (out!=15)
    out[mask_h] = 0 
    out[mask_o] = 255
    # bg = np.repeat(np.expand_dims(out, 2), 3, axis=2) / 255 * np.asarray(input_image)
    # bg = bg.astype("uint8")
    r = Image.fromarray(out).resize(input_image.size)
    r = r.filter(ImageFilter.MinFilter(9))
    output_file = "%s/%s.png" % (mask_path, filename.split("/")[-1]) 
    r.save(output_file)


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles

# get body keypoints drawings
keypoints_dir = f'{data_dir}/keypoints_2d'
keypoints_vis_dir = f'{data_dir}/keypoints_2d_vis'

file_path = sorted(glob.glob(f'{data_dir}/images/*{ref_video}.png'))

subprocess.run(["mkdir", "-p", str(keypoints_dir)])
subprocess.run(["mkdir", "-p", str(keypoints_vis_dir)])

images = {name: cv2.imread(name) for name in file_path}


mp_face_mesh = mp.solutions.face_mesh
# for hand
style_1 = mp_drawing.DrawingSpec(color=(200,0,0), thickness=2, circle_radius=2)
style_2 = mp_drawing.DrawingSpec(color=(200,0,0), thickness=2, circle_radius=2)

# for face landmarks
style_3 = mp_drawing.DrawingSpec(color=(0,128,0), thickness=1, circle_radius=1)
style_4 = mp_drawing.DrawingSpec(color=(255,255,255), thickness=1, circle_radius=1)

# for pose landmarks
style_5 = mp_drawing.DrawingSpec(color=(0,0,255), thickness=3, circle_radius=5)

# Whether to visualize on the original input
vis = True 
save_numpy = False 

face_landmark_list = []
for connection in mp_face_mesh.FACEMESH_CONTOURS:
  face_landmark_list.append(connection[0])
  face_landmark_list.append(connection[1])
face_landmark_list = list(set(face_landmark_list))
face_landmark_list = face_landmark_list[::2]

if save_numpy:
    results_np_list = []


with mp_holistic.Holistic(
    static_image_mode=False, 
    min_detection_confidence=0.5, 
    enable_segmentation=True,
    smooth_segmentation=True,
    model_complexity=2) as holistic:
  for name, image in images.items():
    # Convert the BGR image to RGB and process it with MediaPipe Pose.
    image_name = name.split("/")[-1]
    results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print nose coordinates.
    image_hight, image_width, _ = image.shape
    if results.pose_landmarks:
      print(
        f'Nose coordinates: ('
        f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x * image_width}, '
        f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * image_hight})'
      )

    # Draw pose landmarks.
    print(f'Pose landmarks of {name}:')
    blank_image = np.ones([image_hight, image_width, 3]) * 255
    annotated_image = blank_image.copy()
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.
        get_default_pose_landmarks_style(), 
        connection_drawing_spec=style_5)

    mp_drawing.draw_landmarks(
        annotated_image,
        results.face_landmarks,
        # mp_holistic.FACEMESH_TESSELATION,
        None,
        landmark_drawing_spec=style_3, 
        connection_drawing_spec=style_4)

    # mp_drawing.draw_landmarks(
    #     image=annotated_image,
    #     landmark_list=results.face_landmarks,
    #     connections=mp_face_mesh.FACEMESH_CONTOURS,
    #     landmark_drawing_spec=None,
    #     connection_drawing_spec=mp_drawing_styles
    #     .get_default_face_mesh_contours_style())

    mp_drawing.draw_landmarks(annotated_image, results.left_hand_landmarks, 
                              mp_holistic.HAND_CONNECTIONS, style_1, style_1)
    mp_drawing.draw_landmarks(annotated_image, results.right_hand_landmarks,
                             mp_holistic.HAND_CONNECTIONS, style_2, style_2)
    # save images

    cv2.imwrite(f"{keypoints_dir}/{image_name}", annotated_image)

    point_list = []
    if save_numpy:
        for j, landmark in enumerate(results.face_landmarks.landmark):
            if j % 3 == 0:
                point_list.append([landmark.x, landmark.y]) 

        print(len(point_list))
        for j, landmark in enumerate(results.pose_landmarks.landmark):
            point_list.append([landmark.x, landmark.y]) 

        if results.left_hand_landmarks is not None:
            for j, landmark in enumerate(results.left_hand_landmarks.landmark):
                point_list.append([landmark.x, landmark.y]) 
        else:
            for j in range(20):
                point_list.append([0, 0])
        
        if results.right_hand_landmarks is not None:
            for j, landmark in enumerate(results.right_hand_landmarks.landmark):
                point_list.append([landmark.x, landmark.y]) 
        else:
            for j in range(20):
                point_list.append([0, 0])

        if len(point_list) == 230:
            point_list.append([0, 0])
        results_np = np.stack(point_list, 0)
        print(results_np.shape)
        results_np_list.append(results_np)

    # for gaussian 
    def get_landmark_gaussian(landmarks, variance=150, landmark_list=None):
      gaussian_list = []
      if not type(landmarks) == type(None):
        x, y = np.mgrid[0:image_hight, 0:image_width]
        pos = np.dstack((x, y))
        for i, landmark in enumerate(landmarks.landmark):
          if type(landmark_list) == type(None):
            rv = multivariate_normal([landmark.y * image_hight, 
                                landmark.x * image_width], [[variance, 0], [0, variance]])
            gaussian_image = rv.pdf(pos)
            n = 1 / gaussian_image.max() 
            gaussian_image = gaussian_image * n
            gaussian_list.append(gaussian_image)
          elif i in landmark_list:
            rv = multivariate_normal([landmark.y * image_hight, 
                                landmark.x * image_width], [[variance, 0], [0, variance]])
            gaussian_image = rv.pdf(pos)
            n = 1 / gaussian_image.max() 
            gaussian_image = gaussian_image * n
            gaussian_list.append(gaussian_image)

        return np.stack(gaussian_list, axis=2)
      # in case no hand is detected
      else:
        gaussian_image = np.zeros([image_hight, image_width, 1]) 
        return np.repeat(gaussian_image, 21, axis=2)

    # cv2.imwrite(f"{keypoints_tensor_dir}/{image_name}", gaussian_image * n * 255)

    if vis:
      annotated_image = image.copy()
      condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
      bg_image = np.zeros(image.shape, dtype=np.uint8)
      bg_image[:] = 0 
      annotated_image = np.where(condition, annotated_image, bg_image)
      mp_drawing.draw_landmarks(
          annotated_image,
          results.pose_landmarks,
          mp_holistic.POSE_CONNECTIONS,
          landmark_drawing_spec=mp_drawing_styles.
          get_default_pose_landmarks_style(), 
          connection_drawing_spec=style_5)

      mp_drawing.draw_landmarks(
          annotated_image,
          results.face_landmarks,
          # mp_holistic.FACEMESH_TESSELATION,
          None,
          landmark_drawing_spec=style_3, 
          connection_drawing_spec=style_4)

    #   mp_drawing.draw_landmarks(
    #     image=annotated_image,
    #     landmark_list=results.face_landmarks,
    #     connections=mp_face_mesh.FACEMESH_CONTOURS,
    #     landmark_drawing_spec=None,
    #     connection_drawing_spec=mp_drawing_styles
    #     .get_default_face_mesh_contours_style())

      mp_drawing.draw_landmarks(annotated_image, results.left_hand_landmarks, 
                                mp_holistic.HAND_CONNECTIONS, style_1, style_1)
      mp_drawing.draw_landmarks(annotated_image, results.right_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS, style_2, style_2)
      # save images
      cv2.imwrite(f"{keypoints_vis_dir}/{image_name}", annotated_image)

if save_numpy:
  results_np_all = np.stack(results_np_list, axis=0)
  np.save(f"{data_dir}/keypoints_3d.npy", results_np_all)

