# Project: Can you unscramble a blurry image? 
![image](figs/example.png)

### [Full Project Description](doc/project3_desc.md)

Term: Spring 2019

## Team 5

- Chen, Xinyi xc2464@columbia.edu
- Dubova, Elena ed2801@columbia.edu
- Hu, Xinyi xh2383@columbia.edu
- Ma, Qiaozhen qm2138@columbia.edu
- Xiao, Caihui cx2225@columbia.edu

**Project summary:** In this project, we performed model evaluation and selection for predictive analytics on image data. 

+ Given 3000 images: 1500 low resolution images and their corresponding high resolution images, we extracted the features of these images and used them to train our models. 

+ We created three models for image super resolution. The first model is the baseline model -- Gradient Boosting Machines. Then we built XGBoost model and CNN model to further improve image resolution . 

+ The baseline model has mean psnr = 25.82641. The XGBoost model has mean psnr = xxx and the CNN model has mean psnr = 27.08050. After comparing the performances of XGBoost model and CNN model, our team decided to run the CNN model during the presentation as it costs much less time to run and gives us the best result that we could have.
		
	
**Contribution statement**: ([default](doc/a_note_on_contributions.md)) 


+ Qiaozhen Ma:

+ Xinyi Chen:

+ Elena Dubova:

+ Caihui Xiao: 

+ Xinyi Hu: Designed and wrote the XGBoost model with Caihui Xiao by using R. Trained the XGBoost model and tuned the depths of XGBoost model. Refined the code structures in R. Predicted the whole 1500 low resolution images using XGBoost model and output its mean psnr value. Edited the project summary in README.md.

Following [suggestions](http://nicercode.github.io/blog/2013-04-05-projects/) by [RICH FITZJOHN](http://nicercode.github.io/about/#Team) (@richfitz). This folder is orgarnized as follows.

```
proj/
├── lib/
├── data/
├── doc/
├── figs/
└── output/
```

Please see each subfolder for a README file.
