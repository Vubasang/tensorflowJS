let model;
let IMAGE_WIDTH = 600;


async function loadModel() {
	console.log("model loading mobilenet model kdfah ...");
	loader = document.getElementById("progress-box");
	load_button = document.getElementById("load-button");
	loader.style.display = "block";
	modelName = "mobilenet";
	model = undefined;
	
	model = await tf.loadLayersModel('models/mobilenet/model.json');

	if (typeof model !== "undefined") {
		loader.style.display = "none";
		load_button.disabled = true;		
		load_button.innerHTML = "Loaded Model";
		console.log("model loaded..");
	}
};

function loadImageLocal() {
	console.log("Click into selected file image");
  	document.getElementById("select-file-box").style.display = "table-cell";
  	document.getElementById("predict-box").style.display = "table-cell";
  	document.getElementById("prediction").innerHTML = "Click predict to find my label!";
    renderImage(this.files);
};




function renderImage(file) {
  var reader = new FileReader();
  reader.onload = function(event) {
    let output = document.getElementById('test-image');
  	output.src = reader.result;
  	output.width = IMAGE_WIDTH;
  }
  
  if(event.target.files[0]){
	reader.readAsDataURL(event.target.files[0]);
  }
}

async function predictImage(){
	console.log("Click predict button");
	if (model == undefined) {
		alert("Please load the model first..")
	}
	if (document.getElementById("predict-box").style.display == "none") {
		alert("Please load an image using 'Upload Image' button..")
	}
	
	let image  = document.getElementById("test-image");
	let tensor = preprocessImage(image, modelName);
	let predictions = await model.predict(tensor).data();
	let results = Array.from(predictions)
		.map(function (p, i) {
			return {
				probability: p,
				className: IMAGENET_CLASSES[i]
			};
		}).sort(function (a, b) {
			return b.probability - a.probability;
		}).slice(0, 10);

	document.getElementById("predict-box").style.display = "block";
	document.getElementById("prediction").innerHTML = "MobileNet prediction:<br><br><b>" + results[0].className + "</b>";

	var ul = document.getElementById("predict-list");
	ul.innerHTML = "";

	results.forEach(function (p) {
		console.log(p.className + " " + p.probability.toFixed(5));
		var li = document.createElement("LI");
		li.innerHTML = p.className + ": " + p.probability.toFixed(5);
		ul.appendChild(li);
	})

	var arrayClassName = [];
	for(var i=0; i < results.length; i++){
		arrayClassName.push(results[i].className);
	}

	var arrayProbability = [];
	for(var i=0; i < results.length; i++){
		arrayProbability.push(results[i].probability.toFixed(5));
	}

	/*function square_data(chart){
		var c = document.createElement("canvas");
		var ctx = c.getContext("2d");
		ctx.fillStyle = "#3300CC";
		ctx.font = "20px Georgia";
		ctx.fillText(chart.dataset.data[chart.dataIndex], 150, 60);
		ctx.stroke();
		return c
	}*/

	var xValues = arrayClassName;
	var yValues = arrayProbability;
	
	new Chart("sang", {
		type: "line",
		data: {
		  labels: xValues,
		  datasets: [{
			fill: false,
			lineTension: 0,
			backgroundColor: "rgba(8, 100, 140, 1)",
			borderColor: "rgba(8, 100, 140, 0.5)",
			data: yValues
		  }]
		},
		options: {
			/*elements:{
				"point":{"pointStyle":square_data},
			},*/
		  	legend: {display: false},
        	title: {
                display: true,
                text: 'Probability Analytics Chart',
				fontSize: 30,
				fontColor: "blue"
            },
			plotOptions: {
				line: {
					dataLabels: {
						enabled: true
					},
					enableMouseTracking: false
				}
			},
		  scales: {
			yAxes: [{ticks: {min: 0, max:1, fontSize: 20}}],
			xAxes: [{ticks: {fontSize: 20}}],
		  }
		}
	  });

	  Highcharts.chart('container', {
		credits: {
            enabled: false
        },
		title: {
			display: true,
            text: 'Probability Analytics Chart',
			style: {
                color: 'blue',
				fontSize: '30px',
                fontWeight: 'bold'
            }
		},
		xAxis: {
			categories: arrayClassName,
			labels: {
                style: {
                    color: 'black',
                    fontSize:'18px'
                }
            },
			title: {
				text: 'Predictions',
				style: {
					color: 'black',
					fontSize: '24px',
				}
			}
		},
		yAxis: {
			labels: {
                style: {
                    color: 'black',
                    fontSize:'18px'
                }
            },
			title: {
				text: 'Probability',
				style: {
					color: 'black',
					fontSize: '24px',
				}
			}
		},
		plotOptions: {
			series: {
				dataLabels: {
					enabled: true,
					borderRadius: 5,
					backgroundColor: 'rgba(252, 255, 197, 0.7)',
					borderWidth: 0,
					borderColor: '#AAA',
					y: -6
				}
			}
		},
		series: [{
			showInLegend: false,
		  /*name: 'Predictions',*/
			data: arrayProbability.map(Number),
		}]
	});

	if (typeof predictions !== "undefined"){
		document.getElementById("progress-box").style.display = "none";
	}
}

function preprocessImage(image, modelName) {
	let tensor = tf.browser.fromPixels(image)
		.resizeNearestNeighbor([224, 224])
		.toFloat();

	if (modelName === undefined) {
		return tensor.expandDims();
	} else if (modelName === "mobilenet") {
		let offset = tf.scalar(127.5);
		return tensor.sub(offset)
			.div(offset)
			.expandDims();
	} else {
		alert("Unknown model name..")
	}
}
