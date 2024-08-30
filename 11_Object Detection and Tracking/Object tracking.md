# **OBJECT TRACKING**

## **1. Introduction**

- Object tracking is the task of taking an initial set of object detections, creating a unique ID for each of the initial detections, and then tracking each of the objects as they move around frames in a video, maintaining the assignment of IDs. (Papers With Code)

- In short, object tracking is the process of:
  - Taking an input (i.e., a video stream).
  - Localizing an object in the first frame.
  - Creating a bounding box around the object.
  - And then continuously keeping track of the object as it moves from frame to frame.

- Some requirements of a good object tracking algorithm are:
  - **Robustness**: The algorithm should be able to handle occlusions, motion blur, and low resolution. For example, if an object is occluded by another object or disappears from view for a few frames, the algorithm should be able to handle these cases.
  - **Real-time**: The algorithm should be able to process frames quickly.
  - **Scalability**: The algorithm should be able to handle multiple objects.
  - **Accuracy**: The algorithm should be able to accurately track objects. Ensuring that the object is tracked correctly is important for downstream tasks such as object recognition and object classification.

<div style="position: relative; width: 100%; text-align: center;">
    <iframe width="560" height="315" src="https://www.youtube.com/embed/_i4numqiv7Y?si=HwEO0yi9FDJTFC3O" title="An example of object tracking" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
</div>

- In the above video, you can see an example of tracking vehicles in a video stream. However, what is the difference between object detection and object tracking?
  - **Object detection**: Object detection is the task of detecting objects in an image or video. It involves localizing objects in an image and classifying them. Object detection algorithms output a list of bounding boxes and the class of the object in each bounding box.
  - Based on the bounding boxes and the class of the object, **Object Tracking** is the task of tracking objects as they move from frame to frame in a video. It involves assigning a unique ID to each object and tracking the object as it moves around the video.

## **2. Types of Object Tracking**

- We can divide object tracking algorithms into two main categories: **Single Object Tracking (SOT)** and **Multiple Object Tracking (MOT)**.

### **2.1 Single Object Tracking (SOT)**

- **Single Object Tracking (SOT)** algorithms track a single object in a video. The goal of SOT algorithms is to track a single object as it moves from frame to frame in a video. SOT algorithms are used in applications such as surveillance, video editing, and augmented reality.

- To track a single object, SOT algorithms typically use a bounding box around the object in the first frame and then track the object as it moves from frame to frame.

<div style="position: relative; width: 100%; text-align: center;">
    <img src="https://i.imgur.com/ASWx8ze.gif" width="400" alt="Single Object Tracking">
</div>

### **2.2 Multiple Object Tracking (MOT)**

- **Multiple Object Tracking (MOT)** algorithms track multiple objects in a video. The goal of MOT algorithms is to track multiple objects as they move from frame to frame in a video. MOT algorithms are used in applications such as traffic monitoring, crowd analysis, and object recognition. Because of that, MOT algorithms are more complex than SOT algorithms and receive more attention in the research community.

<div style="position: relative; width: 100%; text-align: center;">
    <img src="https://i.imgur.com/qdl714G.gif" width="400" alt="Multiple Object Tracking">
</div>

- MOT is also divided in many ways, such as:
  - **Online MOT**: process frames in real-time and do not have access to future frames, it only uses the current and past frames to track objects. This makes online MOT algorithms suitable for real-time applications where processing speed is important but may result in less accurate tracking. Online MOT algorithms are used in applications where real-time processing is required, such as surveillance and traffic monitoring.
  - **Offline MOT**: Offline MOT algorithms process frames in batch mode and have access to future frames, so it will have better tracking accuracy. Offline MOT algorithms are used in applications where tracking accuracy is more important than real-time processing, such as video editing and object recognition.

## **3. Dataset and Evaluation Metrics**

- Some popular datasets for object tracking can be listed as: **MOT Challenge**, **ImageNetVID**, **KITTI**,...
- To evaluate object tracking, we can use the following metrics:
  - FP (False Positive): number of frames where the tracker incorrectly detects an object.
  - FN (False Negative): number of frames where the tracker fails to detect an object.
  - ID Switch: number of times the tracker switches the ID of an object.
  - MOTA (Multiple Object Tracking Accuracy): a metric that combines FP, FN, and ID Switch to evaluate the overall tracking performance. The higher the MOTA, the better the tracking performance.

  $$MOTA = 1 - \frac{\sum_{t} (FN_t + FP_t + IDsw_t)}{\sum_{t} GT_t}$$

  - MOTP (Multiple Object Tracking Precision): a metric that evaluates the localization accuracy of the tracker. The higher the MOTP, the better the localization accuracy.

    $$MOTP = \frac{\sum_{t} \sum_{i} d_{i,t}}{\sum_{t} c_t}$$

    where: $d_{i,t}$ is the distance between the predicted bounding box and the ground truth bounding box for object $i$ at time $t$, and $c_t$ is the number of objects at time $t$.

  - **Hz or FPS**: the number of frames processed per second. The higher the Hz or FPS, the faster the tracker.
  - **HOTA (Higher Order Tracking Accuracy)**: a metric that evaluates the tracking performance of the tracker. HOTA is a more comprehensive metric than MOTA and MOTP and takes into account the higher-order relationships between objects.
  - **DetA (Detection Accuracy)**: a metric that evaluates the detection performance of the tracker. DetA is a more comprehensive metric than MOTA and MOTP and takes into account the detection performance of the tracker.
  - **AssA (Association Accuracy)**: a metric that evaluates the association performance of the tracker. AssA is a more comprehensive metric than MOTA and MOTP and takes into account the association performance of the tracker.