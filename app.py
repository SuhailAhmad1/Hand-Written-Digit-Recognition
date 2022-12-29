import gradio as gr
     
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

#Getting the Data
print("#Getting the data")
Train_data = pd.read_csv("./dataset/mnist_train.csv") 
x_train = Train_data.iloc[:,1:].values
y_train = Train_data.label

Test_data = pd.read_csv("./dataset/mnist_test.csv")
x_test = Test_data.iloc[:,1:].values
y_test = Test_data.label

#Preprocessing of Data
print("#Vector conversion and Normalizing")
x_train = x_train.reshape(x_train.shape[0], 28*28)
x_test = x_test.reshape(x_test.shape[0], 28*28)

x_train = x_train/255
x_test = x_test/255


#Training Random Forest Classifier
print("#Training the Model")
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)

#Evaluating the Model
print("#Evaluating the Model")
print(rfc.score(x_test, y_test))

values = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
#Function for Gradio Interface
def predict_image(img):
    img = img.reshape(-1, 28*28)
    img = img/255

    prediction=rfc.predict(img.reshape(1,-1))[0]
    
    print(prediction)
    return values[prediction]

     

#GUI Using gradio
iface = gr.Interface(
    predict_image,
    inputs="sketchpad", 
    outputs="label",
    title="Handwritten Digit Recognition",
    allow_flagging="never",
    description="Project Done By : Suhail, Gowhar and Sayar",
    css="""
        .gradio-container {
            background-color:grey
        }
        #component-5 {
            justify-content: center;
            align-items: center;
            flex-direction: column;
        }
        h1 {
            color: azure;
            font-weight: 600;
            font-size: 3em;
            margin-top: 0;
            margin-bottom: .8888889em;
            line-height: 1.1111111;
        }
        .gr-prose p {
            color: azure;
            text-align: center
        }
        .gr-panel {
            background-color: rgb(82 91 95 / 1)
        }
    """
    )

iface.launch(debug='True')