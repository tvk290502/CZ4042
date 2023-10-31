import React from 'react';
import Plot from 'react-plotly.js';
// import d from "/Users/vinhkhaitruong/Documents/CE4045/npm_project/src/NLP_preprocessing/sample.json"
const d = {
'the': 15893,
 'to': 10574,
 'is': 9921,
 'and': 8981,
 'of': 7320,
 'i': 6764,
 'you': 5705,
 'are': 5695,
 'it': 5381,
 'in': 5346
}

let values = []
let labels = []

for(var e in d){
	labels.push(e)
	values.push(d[e])
}

const BarChartForStopWords = (props) =>{

	// let type = props.onGetType()

	// let name = props.onGetName()

	
	var data = [{
		type: 'bar',
		x: values,
		y: labels,
		orientation: 'h'
	      }];

	

	return(
	<div>

		<Plot 
			data = {data}
			layout={ {width: 600, height: 450, title: "10 Most Common Stop Words",paper_bgcolor:'FBFBF9',plot_bgcolor:'FBFBF9' }}
		/>
	</div>
	);
}

export default BarChartForStopWords;