# support-chatbot

notebooks - jupyter notebooks for EDA, training and testing

src - main code

data - examples of datasets

### backlog:
https://miro.com/app/board/uXjVPlKl-Pk=/?share_link_id=494559570972

### analysis on transcribed texts by the model'golos_ft_50_epoch_with_specAug'
The model recognizes digits and numbers very well. Makes mistakes in named entities. Recognizes the word "удостоверение" as "удовлетворение". It recognizes the speech of the bot from Audio3 very well (apparently, due to the fact that all sounds are pronounced clearly by the bot). Does not understand the word "доска". Does not recognize very quiet speech at all. In general, It does not understand where the beginning and the end of the word are in the audio recordings with noises. Poorly recognizes the words in which some sounds are swallowed and indistinctly pronounced. There are a few examples of transcribed texts below.

True: 
![image](https://user-images.githubusercontent.com/113451350/228051151-e923aed7-2490-47f0-be9a-8ddf32db8f1d.png)
Predicted without augmentation: 
![image](https://user-images.githubusercontent.com/113451350/228051316-448b1f5d-c44f-47ca-83f0-7488fe1c8808.png)
Predicted with augmentation:
![image](https://user-images.githubusercontent.com/113451350/228052469-646e0a6f-9f47-4498-8d65-2cf0e36d53ce.png)

True: 
![image](https://user-images.githubusercontent.com/113451350/228053209-544ae0d7-aae3-47cd-a7e5-dd2e39d963d9.png)

Predicted without augmentation:
![image](https://user-images.githubusercontent.com/113451350/228053422-d50694fb-60a7-4fe1-a814-d6c99653e926.png)

Predicted with augmentation: 
![image](https://user-images.githubusercontent.com/113451350/228054195-be3e7988-de0c-4bfb-828f-d5328efd8a0e.png)

Results: WER = 0.739, CER = 0.392 

