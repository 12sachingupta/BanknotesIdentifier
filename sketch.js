
var Camera;
var AllTrainingButtons;
var knn;
var Model;
var Text;
var Classifying = false;
var InputTexbox;
var ButtonTexBox;
var result_label_aux = "";

function setup() {
  var txt = createP('Cash could be confusing for blind people since the banknotes are similar to each other in shape, size and weight. Blind people could have problems in stores buying and receiving change or withdraw money from ATM, they depend on the kindness and honesty of strangers to advise them of the amount of money they have in their hands. But technology advances are helping to blind people to be more independent, for example, they have electronic portable identifiers to recognize money.<br>Banknotes Identifier is a talking application that consists of a Machine Learning model training to recognize 10,20,50,100 ₹ banknotes. You just have to put a banknote in front of the camera and a voice will tell you which banknote it is.');
  txt.style("text-align", "justify");

  createCanvas(320, 240);
  background(255, 0, 0);
  Camera = createCapture(VIDEO);
  Camera.size(320, 240);
  Camera.hide();

  Model = ml5.featureExtractor('MobileNet', ReadyModel);
  knn = ml5.KNNClassifier();

  txt = createP('To train the model you have to put a banknote in front of the camera and click on the button according to the banknote.<br>Press buttons to train the model');
  txt.style("text-align", "center");
  var note_5 = createButton("10₹");
  note_5.class("TrainingButton");

  var note_10 = createButton("20₹");
  note_10.class("TrainingButton");

  var note_20 = createButton("50₹");
  note_20.class("TrainingButton");

  var note_50 = createButton("100₹");
  note_50.class("TrainingButton");

  var nothing = createButton("Nothing");
  nothing.class("TrainingButton");


  txt = createP('To save your trained model you have to click on the "Save" button and a file will download automatically, that is your trained model!<br>And to load your previous saved trained model you have to click on the "Load" button and the last model file that you have saved will load!');
  txt.style("text-align", "center");
  var SaveButton = createButton("Save");
  SaveButton.mousePressed(SaveModel);
  var LoadButton = createButton("Load");
  LoadButton.mousePressed(LoadModel);

  Text = createP("The Model is not ready. Waiting...");
  Text.style("text-align", "center");
  AllTrainingButtons = selectAll(".TrainingButton");

  for (var B = 0; B < AllTrainingButtons.length; B++) {
    AllTrainingButtons[B].style("margin-left", "80px");
    AllTrainingButtons[B].style("padding", "6px");
    AllTrainingButtons[B].mousePressed(PressingButton);
  }
  SaveButton.style("margin-left", "480px");
  SaveButton.style("padding", "6px");
  LoadButton.style("margin-left", "15px");
  LoadButton.style("padding", "6px");

  speechSynthesis.getVoices().forEach(function(voice) {
     console.log('Hi! My name is ', voice.name);
  });

}

function PressingButton() {
  var ButtonName = this.elt.innerHTML;
  console.log("Training " + ButtonName);
  knnTraining(ButtonName);
}

function knnTraining(TrainingObject) {
  const Image = Model.infer(Camera);
  knn.addExample(Image, TrainingObject);
}

function ReadyModel() {
  console.log("The Model is done");
  Text.html("The Model is done");
  Text.style("text-align", "center");
  Text.style("font-size","24px");
}

function classifier() {
  const Image = Model.infer(Camera);
  knn.classify(Image, function(error, result) {
    if (error) {
      console.error();
    } else {
      Text.html("This is " + result.label);
      Text.style("text-align", "center");
      Text.style("font-size","24px");
      //classifier();
      if(result.label != result_label_aux){
        result_label_aux = result.label;
            SayIt("This is " + result.label);
      }
    }
  });
}

function SayIt(Text){
  var utterance  = new SpeechSynthesisUtterance();
  var voice = speechSynthesis.getVoices().findIndex(function (voice) {
    return voice.name === 'Google UK English Female';
  });
  utterance.voice = speechSynthesis.getVoices()[voice];
utterance.text = Text;
speechSynthesis.speak(utterance);

}

function TraningTextBox() {
  const Image = Model.infer(Camera);
  knn.addExample(Image, InputTexbox.value());
}

function SaveModel() {
  if (Classifying) {
    save(knn, "Model.json");
  }
}

function LoadModel() {
  console.log("Loading the model...");
  knn.load("./Model.json", function() {
    console.log("Model loaded");
    Text.html("Model loaded");
    Text.style("text-align", "center");
    Text.style("font-size","24px");
  });
}

function draw() {
  image(Camera, 0, 0, 320, 240);

  if (knn.getNumLabels() > 0 && !Classifying) {
    //classifier();
    setInterval(classifier, 500);
    Classifying = true;
  }
}


const save = (knn, name) => {
  const dataset = knn.knnClassifier.getClassifierDataset();
  if (knn.mapStringToIndex.length > 0) {
    Object.keys(dataset).forEach(key => {
      if (knn.mapStringToIndex[key]) {
        dataset[key].label = knn.mapStringToIndex[key];
      }
    });
  }
  const tensors = Object.keys(dataset).map(key => {
    const t = dataset[key];
    if (t) {
      return t.dataSync();
    }
    return null;
  });
  let fileName = 'myKNN.json';
  if (name) {
    fileName = name.endsWith('.json') ? name : `${name}.json`;
  }
  saveFile(fileName, JSON.stringify({
    dataset,
    tensors
  }));
};

const saveFile = (name, data) => {
  const downloadElt = document.createElement('a');
  const blob = new Blob([data], {
    type: 'octet/stream'
  });
  const url = URL.createObjectURL(blob);
  downloadElt.setAttribute('href', url);
  downloadElt.setAttribute('download', name);
  downloadElt.style.display = 'none';
  document.body.appendChild(downloadElt);
  downloadElt.click();
  document.body.removeChild(downloadElt);
  URL.revokeObjectURL(url);
};
