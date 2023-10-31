import * as React from 'react';
import { TextField, Button } from '@mui/material';
import { useRef, useState } from 'react';
import * as constants from "/Users/vinhkhaitruong/Documents/CE4045/npm_project/src/Constants.js";


const axios = require('axios')


const InputBarForBert = (props) =>{

	
	
	const [textInput, setTextInput] = useState('');

	const handleTextInputChange = event => {
	    setTextInput(event.target.value);
	};

	const textFieldRef = useRef();

	function firstFunction(){
		console.log("testing")
	}

	
      async function readTextFieldValue (){
	
	axios.post('http://localhost:5005/test',{'input':textFieldRef.current.value}).then(
		res =>{
			props.onPassBertResult(res.data.result)
			console.log(res.data.tokenize)
			props.onPassTokenize(res.data.tokenize)
			// console.log(res.data.result)
		
		}
	)

	// axios.get('http://localhost:5000/getTest').then(
	// 	res => {
	// 		console.log(res.data.result)
	// 	}
	// )
	      
}

 
    return (
<div>
	<div>
	<TextField fullWidth  id="outlined-basic" label="Text Input" variant="outlined" onChange= {handleTextInputChange} inputRef={textFieldRef} />
	</div>

	<div>
	<Button onClick = {readTextFieldValue} variant="contained">Predict Sentiment</Button>
	</div>

</div>
    );
}

export default InputBarForBert;
