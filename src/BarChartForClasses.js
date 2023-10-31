import React from 'react';
import Plot from 'react-plotly.js';
// import d from "/Users/vinhkhaitruong/Documents/CE4045/npm_project/src/NLP_preprocessing/sample.json"
const d = {
	'normal':[8827,7285,3651],
	'balance':[8827,8827,8827]
}
const BarChartForClass = (props) =>{

	let type = props.onGetType()

	let name = props.onGetName()

	
	var data = [{
		type: 'bar',
		x: d[type],
		y: ['Negative','Neutral','Positive'],
		orientation: 'h'
	      }];

	

	return(
	<div>

		<Plot 
			data = {data}
			layout={ {width: 600, height: 300, title: name,paper_bgcolor:'FBFBF9',plot_bgcolor:'FBFBF9' }}
		/>
	</div>
	);
}

export default BarChartForClass;