# Project Summary & Report Notes

## Milestone 1: Initial Setup & Challenges

* **Approach:** Fine-tuning a pre-trained model on a custom dataset.
* **Constraints:** Training performed locally on laptops to ensure data privacy.
* **Labelling:** Manual labelling using Label Studio (Classes: Dirt, Dirt-Wire, Burn, Open) was performed since no pre-labelled data was available to us.
* **Challenge:** Poor generalization due to multiple colors and different chip layouts.
* *Attempt 1:* Data Augmentation.
* *Attempt 2:* Labelling specific layouts separately.

* **Conclusion M1:** The task was too complex for the amount of data that we would have to label, and defect definitions were ambiguous. A meeting was scheduled to clarify priorities.

## Milestone 2: Scope Refinement & SAHI Integration

* **Scope Change:** Simplified to 2 classes: "Critical" (on wire) and "Non-Critical" (outside wire). Relabelling was performed accordingly again using Label Studio.
* **SAHI Implementation:**
* Adopted Slicing Aided Hyper Inference (SAHI) to detect small defects since the previous approach where the model processed the entire image at once was ineffective because the receptive field was too large compared to the defect size.
* Created preprocessing scripts to convert datasets to COCO format (required by SAHI).


* **Fine-Tuning Experiments:**
* Compared partial vs. full fine-tuning.
* *Result:* Partial training is viable for quick iteration, but full fine-tuning yields the best results.


* **Class Balancing:**
* Addressed class imbalance (#Non-Critical > #Critical).
* *Tested Downsampling:* Removing non-critical samples (discarded too much manual work that we have already done ourselves in labelling).
* *Tested Oversampling:* Duplicating critical samples with augmentation to prevent overfitting.
* *Decision:* **Oversampling** was chosen to maximize data usage since the dataset was already small.


* **Conclusion M2:** Performance improved but plateaued. The root cause was identified as poor data quality/labeling rather than model architecture. Why? Because the labels were inconsistent, with varying bounding box sizes and unclear definitions of some defects. A final relabelling effort was done in the next milestone.

## Milestone 3: Final Relabelling & Optimization

* **Final Relabelling:**
* We redefined the classes to "Critical" and "Dirt-Wire," discarding the "Non-Critical" category entirely as it was deemed irrelevant by the lab and disproportionately time-consuming to label all of them again. With this, we focused solely on defects that matter making the model more useful in practice. 
* *Focus:* Consistent bounding box sizes and precise fit around objects.


* **Optimization Tweaks:**
* **Slice Size**: We initially tested 128px slices to maximize detection of small defects but observed high false positives due to insufficient background context. We settled on **256px**, which provided the optimal balance—retaining sensitivity to small defects while offering enough context to significantly improve precision and reduce false positives.



* **Resolution (`imgz`):** Increased from 600 (ad-hoc) to **768** based on documentation formulas and the size of normal defects in our images, enhancing feature representation without excessive computational cost.
* **Background Images:** We incorporated **50%** of the available background-only images (defect-free chips) to help the model distinguish defects from normal features, striking a balance between improved accuracy and training time efficiency.


### Final Results (Slice 256, Imgz 768, Bg 50%)

* **COCO Metrics (global):**
* mAP @ 0.50: **0.323**
* mAP @ 0.50:0.95: **0.126**


* **Mean F1 Score:** 0.497

| Class | Precision | Recall | F1-Score |
| --- | --- | --- | --- |
| **Critical** | 0.435 | 0.435 | 0.435 |
| **Dirt-Wire** | 0.674 | 0.478 | 0.560 |

* **Conclusion M3:**
* **Perfect Classification:** Zero confusion between classes (Critical is never predicted as Dirt-Wire and vice versa).
* **Recall Issues:** The model still misses some defects.
* **Data Impact:** Label consistency done in this Milestone significantly improved performance (in the whole project were 15h total labeling time invested). We strongly believe more labelled data of similar quality would further boost results. In the YOLO documentation they say we need 10k of labelled bounding boxes per class to reach good performance, while we have ~100x less than that for Critical class that is the less represented.

## Extras: Tools Developed

* **Lab Interface:** A GUI for the lab to upload images and view model predictions instantly.
* **Auto-Labelling:** Developed a Machine Learning backend connecting the model to Label Studio, accelerating the labeling process.

## Limitations & Report Context

* **Dataset Size:** We possess ~100x less data than the recommended amount per class (10k images) mentioned in YOLO documentation. This is the primary bottleneck for performance.
* **Compute Resources:** Relied on fine-tuning pre-trained models on local GPUs due to privacy requirements. We did not follow the approach of training a model from scratch on large-scale datasets since we did not have access to such resources.
* **Architecture:** Mention the YOLO architecture choice (parameters vs. model size trade-offs).
