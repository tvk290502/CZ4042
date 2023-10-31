import React from 'react';
import Plot from 'react-plotly.js';
import d from "/Users/vinhkhaitruong/Documents/CE4045/npm_project/src/NLP_preprocessing/sample.json"

const PieChart = () =>{

	let values = []
	let labels = []
	for(var e in d){
		values.push(d[e])
		labels.push(e)
	}

	var data = [{
		type: 'pie',
		values: values,
		labels: labels,
	      }];

	return(
	<div>
		<Plot 
			data = {data}
			layout={ {width: 600, height: 600, title: 'Pie Chart',legend:{orientation:'h', bgcolor:'#F3EAD3'},plot_bgcolor:'FBFBF9',paper_bgcolor:'FBFBF9' }}
		/>
	</div>
	);
}

export default PieChart 