# edge-wake-engine
A cascading architecture for wake word detection using classical SP and DL.

## Data Sources

This project utilizes the following open-source acoustic datasets for model training and validation:

* **[Google Speech Commands Dataset (v2)](http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz)**
  * **Usage:** Provides the positive target wake words and negative human speech (non-target words). All audio is pre-formatted as 1-second, 16kHz `.wav` files.
  * **Citation:** Warden, P. (2018). Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition. *arXiv preprint arXiv:1804.03209*.

* **[ESC-50: Dataset for Environmental Sound Classification](https://github.com/karolpiczak/ESC-50)**
  * **Usage:** Serves as hard negative examples (environmental noise, HVAC, sirens, keyboard typing) to train the Stage 1 CNN against false positives. Also utilized for synthetic background noise mixing during DSP data augmentation.
  * **Citation:** Piczak, K. J. (2015). ESC: Dataset for Environmental Sound Classification. *Proceedings of the 23rd Annual ACM Conference on Multimedia*.
