
<script>
window.onload = function () {

//Better to construct options first and then pass it as a parameter
var options = {
	animationEnabled: true,
	title: {
		text: "Rating of Bullies",
		fontColor: "Peru"
	},
	axisY: {
		tickThickness: 0,
		lineThickness: 0,
		valueFormatString: " ",
		gridThickness: 0
	},
	axisX: {
		tickThickness: 0,
		lineThickness: 0,
		labelFontSize: 18,
		labelFontColor: "Peru"
	},
	data: [{
		indexLabelFontSize: 26,
		toolTipContent: "<span style=\"color:#62C9C3\">{indexLabel}:</span> <span style=\"color:#CD853F\"><strong>{y}</strong></span>",
		indexLabelPlacement: "inside",
		indexLabelFontColor: "white",
		indexLabelFontWeight: 600,
		indexLabelFontFamily: "Verdana",
		color: "#62C9C3",
		type: "bar",
		dataPoints: [
			{ y: 21, label: "10%", indexLabel: "Magazines" },
			{ y: 25, label: "25%", indexLabel: "Social" },
			{ y: 33, label: "29%", indexLabel: "Entertainment" },
			{ y: 36, label: "30%", indexLabel: "News" },
			{ y: 42, label: "32%", indexLabel: "HelloApp" },
			{ y: 49, label: "39%", indexLabel: "Whatsapp" },
			{ y: 50, label: "50%", indexLabel: "Instagram" },
			{ y: 55, label: "65%", indexLabel: "Facebook" },
			{ y: 61, label: "81%", indexLabel: "Twitter" }
		]
	}]
};

$("#chartContainer").CanvasJSChart(options);
}
</script>
</head>


 <div class="content">
            <div class="container-fluid">
                <div class="row">
                    <div class="col-md-12">
                        <div class="card">

                            <div class="header">
<h4 class="title">MONTHLY ANALYSIS</h4>
                                <p class="category">Last Campaign</p>
                            </div>

                            <div class="content">
                              <script>
window.onload = function () {

var options = {
	animationEnabled: true,
	theme: "light2",
	title:{
		text: "Use of Twitter Comparison by Location"
	},
	axisY: {
		title: "Uses in month",
		valueFormatString: "#0",
		suffix: "K",
		prefix: "£"
	},
	legend: {
		cursor: "pointer",
		itemclick: toogleDataSeries
	},
	toolTip: {
		shared: true
    },
	data: [{
		type: "area",
		name: "London",
		markerSize: 5,
		showInLegend: true,
		xValueFormatString: "MMMM",
		yValueFormatString: "£#0K",
		dataPoints: [
			{ x: new Date(2017, 0), y: 12 },
			{ x: new Date(2017, 1), y: 15 },
			{ x: new Date(2017, 2), y: 12 },
			{ x: new Date(2017, 3), y: 17 },
			{ x: new Date(2017, 4), y: 20 },
			{ x: new Date(2017, 5), y: 21 },
			{ x: new Date(2017, 6), y: 24 },
			{ x: new Date(2017, 7), y: 19 },
			{ x: new Date(2017, 8), y: 22 },
			{ x: new Date(2017, 9), y: 25 },
			{ x: new Date(2017, 10), y: 21 },
			{ x: new Date(2017, 11), y: 19 }
		]
	}, {
		type: "area",
		name: "India",
		markerSize: 5,
		showInLegend: true,
		yValueFormatString: "Rs#0K",
		dataPoints: [
			{ x: new Date(2017, 0), y: 8 },
			{ x: new Date(2017, 1), y: 12 },
			{ x: new Date(2017, 2), y: 9 },
			{ x: new Date(2017, 3), y: 11 },
			{ x: new Date(2017, 4), y: 15 },
			{ x: new Date(2017, 5), y: 12 },
			{ x: new Date(2017, 6), y: 13 },
			{ x: new Date(2017, 7), y: 9 },
			{ x: new Date(2017, 8), y: 7 },
			{ x: new Date(2017, 9), y: 14 },
			{ x: new Date(2017, 10), y: 18 },
			{ x: new Date(2017, 11), y: 14 }
		]
	}]
};
$("#chartContainer1").CanvasJSChart(options);

function toogleDataSeries(e) {
	if (typeof (e.dataSeries.visible) === "undefined" || e.dataSeries.visible) {
		e.dataSeries.visible = false;
	} else {
		e.dataSeries.visible = true;
	}
	e.chart.render();
}

}
</script>
                                <div id="chartContainer1" style="height: 300px; width: 100%;"></div>
<script src="https://canvasjs.com/assets/script/jquery-1.11.1.min.js"></script>
<script src="https://canvasjs.com/assets/script/jquery.canvasjs.min.js"></script>


