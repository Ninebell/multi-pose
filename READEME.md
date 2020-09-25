# Multi Pose Estimation With Attention to Center

----
### What I will do in this repository

- Training   
    - StackedHourglass.
    - High-Resolution Networks.

- Evaluation
    - StackedHourglass.
    - High-Resolution Networks.

### What I already done.
   
- Dataset
    - MPII: I used MPII Dataset and I implemetend code to mpii form to heatmap and affinity field.
        - input
            - image
        - label
            - joint -> heatmap  
            - limb - > affinity field ( point from center to joint)


- Inference
    - Center Attention
    > My method were different with OpenPose. I just used center point about person.  
      so my affinity field actually point from center to joint. and calculate energy between center and joint.
    
    


