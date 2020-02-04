# ML-Toy
A small application in which you can play with very simple machine learning tasks.  
I hope it will help you understand better how machine learning models work.  

It allows you to create a simple 2D dataset and choose one of many sklearn models:
![Screenshot 1](screenshots/Screenshot_1.png)
You can play with model parameters and see what happens:
![Screenshot 1](screenshots/Screenshot_2.png)

## Try it online
App is deployed here: https://ml-toy.herokuapp.com/

## Run locally
```
sudo pip install streamlit
git clone https://github.com/hocop/ML-Toy
streamlit run ml_toy.py
```
It will start running on localhost. The browser window should open automaticaly.  
`Procfile` and `setup.sh` are only for heroku, normally you don't need them.
