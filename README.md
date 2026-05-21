# CodeAlpha_Speech_Emotion
# Task 2: Speech Emotion Recognition Engine using 1D CNN

Proud to share Task 2 of my Machine Learning Internship at CodeAlpha. This project focuses on processing raw speech audio signals to accurately classify underlying human emotions using Deep Learning.

---

##  Complete Technical Flow Diagram

The engineering architecture transforms continuous analog audio time-domain variables into distinct emotional probability weights via this precise pipeline:

```text
 [ Raw Waveform Audio (.wav) ]  <-- Input Signal (Time Domain)
               │
               ▼
   [ Librosa Preprocessing ]     <-- Sound sampling, duration trimming
               │
               ▼
    [ Feature Extraction ]       <-- Librosa.feature.mfcc()
               │
               ▼
 [ 40 MFCC Coefficients Matrix ] <-- Transformed into Matrix Array
               │
               ▼
      [ Mean Temporal Pool ]     <-- Averaged via np.mean(axis=0)
               │
               ▼
   [ 1D CNN Feature Learning ]   <-- Spatial patterns extracted across frequencies
               │
               ▼
   [ Dense / Softmax Classification ] <-- Probability distribution (Happy vs Sad vs Angry)
               │
               ▼
       [ Gradio Web UI ]         <-- Final User-Interface Deployment Output
