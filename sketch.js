let Camera;
let AllTrainingButtons;
let knn;
let Model;
let Text;
let Classifying = false;
let result_label_aux = "";

function setup() {
    let txt = createP('Cash could be confusing for blind people since the banknotes are similar to each other in shape, size, and weight. Blind people could have problems in stores buying and receiving change or withdrawing money from ATMs. They depend on the kindness and honesty of strangers to advise them of the amount of money they have in their hands. But technology advances are helping blind people to be more independent. For example, they have electronic portable identifiers to recognize money. Banknotes Identifier is a talking application that uses a Machine Learning model to recognize 10, 20, 50, and 100 ₹ banknotes. Just put a banknote in front of the camera, and a voice will tell you which banknote it is.');
    txt.style("text-align", "justify");

    createCanvas(320, 240);
    background(255, 0, 0);
    Camera = createCapture(VIDEO);
    Camera.size(320, 240);
    Camera.hide();

    Model = ml5.featureExtractor('MobileNet', ReadyModel);
    knn = ml5.KNNClassifier();

    txt = createP('To train the model, put a banknote in front of the camera and click the corresponding button.<br>Press buttons to train the model');
    txt.style("text-align", "center");

    createTrainingButtons();

    txt = createP('To save your trained model, click the "Save" button. To load your previously saved model, click the "Load" button.');
    txt.style("text-align", "center");
    
    let SaveButton = createButton("Save");
    SaveButton.mousePressed(SaveModel);
    
    let LoadButton = createButton("Load");
    LoadButton.mousePressed(LoadModel);

    Text = createP("The Model is not ready. Waiting...");
    Text.style("text-align", "center");
    AllTrainingButtons = selectAll(".TrainingButton");

    styleButtons(AllTrainingButtons, SaveButton, LoadButton);

    speechSynthesis.getVoices().forEach(voice => console.log('Hi! My name is ', voice.name));
}

function createTrainingButtons() {
    const labels = ["10₹", "20₹", "50₹", "100₹", "Nothing"];
    labels.forEach(label => {
        let btn = createButton(label);
        btn.class("TrainingButton");
        btn.mousePressed(() => knnTraining(label));
    });
}

function styleButtons(trainingButtons, saveButton, loadButton) {
    trainingButtons.forEach(btn => {
        btn.style("margin-left", "80px");
        btn.style("padding", "6px");
    });
    saveButton.style("margin-left", "480px");
    saveButton.style("padding", "6px");
    loadButton.style("margin-left", "15px");
    loadButton.style("padding", "6px");
}

function knnTraining(label) {
    const Image = Model.infer(Camera);
    knn.addExample(Image, label);
}

function ReadyModel() {
    console.log("The Model is done");
    Text.html("The Model is done");
    Text.style("text-align", "center");
    Text.style("font-size", "24px");
}

function classifier() {
    const Image = Model.infer(Camera);
    knn.classify(Image, (error, result) => {
        if (error) {
            console.error(error);
            return;
        }
        Text.html("This is " + result.label);
        Text.style("text-align", "center");
        Text.style("font-size", "24px");
        if (result.label !== result_label_aux) {
            result_label_aux = result.label;
            SayIt("This is " + result.label);
        }
    });
}

function SayIt(text) {
    let utterance = new SpeechSynthesisUtterance();
    let voice = speechSynthesis.getVoices().find(v => v.name === 'Google UK English Female');
    utterance.voice = voice;
    utterance.text = text;
    speechSynthesis.speak(utterance);
}

function SaveModel() {
    if (Classifying) {
        save(knn, "Model.json");
    }
}

function LoadModel() {
    console.log("Loading the model...");
    knn.load("./Model.json", () => {
        console.log("Model loaded");
        Text.html("Model loaded");
        Text.style("text-align", "center");
        Text.style("font-size", "24px");
    });
}

function draw() {
    image(Camera, 0, 0, 320, 240);

    if (knn.getNumLabels() > 0 && !Classifying) {
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
        return t ? t.dataSync() : null;
    });
    const fileName = name.endsWith('.json') ? name : `${name}.json`;
    saveFile(fileName, JSON.stringify({ dataset, tensors }));
};

const saveFile = (name, data) => {
    const downloadElt = document.createElement('a');
    const blob = new Blob([data], { type: 'octet/stream' });
    const url = URL.createObjectURL(blob);
    downloadElt.setAttribute('href', url);
    downloadElt.setAttribute('download', name);
    downloadElt.style.display = 'none';
    document.body.appendChild(downloadElt);
    downloadElt.click();
    document.body.removeChild(downloadElt);
    URL.revokeObjectURL(url);
};
