import React from 'react';
import Plot from 'react-plotly.js';
import d from "/Users/vinhkhaitruong/Documents/CE4045/npm_project/src/NLP_preprocessing/sample.json"

const Histogram = () =>{

	let values = []
	let labels = []
	for(var e in d){
		values.push(d[e])
		labels.push(e)
	}

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
			layout={ {width: 600, height: 600, title: 'Bar Chart',paper_bgcolor:'FBFBF9',plot_bgcolor:'FBFBF9' }}
		/>
	</div>
	);
}

export default Histogram 