import logo from './logo.svg';
import './App.css';
import InputBar from './inputBar';
import Histogram from './Histogram';
import PieChart from './PieChart';
import {React, useEffect, useState} from 'react'
// import axios, * as others from 'axios';
import myData from '/Users/vinhkhaitruong/Documents/CE4045/npm_project/src/NLP_preprocessing/sample.json';
import BarChartForClass from './BarChartForClasses';
import InputBarForBert from './inputBarForBert';
import BarChartForStopWords from './BarChartForStopWords';

let stop_words = ['the',
'to',
'and',
'is',
'of',
'i',
'it',
'in',
'you',
'are',
'that',
'’',
'for',
'people',
'they',
'this',
'with',
'be',
'on',
'have',
's',
'about',
'so',
'we',
'or',
'if',
'who',
't',
'was',
'an',
'from',
'can',
'their',
'my',
'what',
'at',
'by',
'your',
'am',
'being',
'no',
'them',
'will',
'me',
'he',
'more',
'because',
'has',
'out',
'would',
'there',
'when',
'how',
'why',
'get',
'think',
'one',
'...',
'up',
'want',
'️',
'know',
'our',
'she',
'also',
'us',
'some',
'other',
'been',
'see',
'women',
'her',
'say',
'does',
'any',
'make',
'then',
'were',
'going',
'its',
're',
'…',
'way',
'time',
'did',
'these',
'2',
'children',
'where',
'm',
'those',
'part',
'said',
'need',
'here',
'go',
'which',
'his',
'into',
'something',
'men',
'3',
'person',
'had',
'thing',
'"',
'things',
'.',
'group',
'got',
'first',
'could',
'someone',
'characters',
'saying',
"that's",
'made',
'show',
'u',
'trying',
'..',
'take',
'school',
'man',
',',
'let',
'after',
'ppl',
'etc',
'2022',
'life',
'live',
'today',
'world',
'him',
'day',
'years',
'book',
'feel',
'use',
'look',
'stuff',
'media',
'under',
'while',
'point',
'read',
'again',
'country',
'since',
'getting',
"there's",
'doing',
'such',
'come',
'making',
've',
'friends',
'having',
'without',
'own',
'stonewall',
'members',
'work',
'makes',
'better',
'‘',
'state',
'before',
'tell',
'family',
'twitter',
'around',
'woman',
'fact',
'give',
'find',
'groups',
'reason',
'character',
'using',
'keep',
'folks',
'via',
'both',
'call',
'two',
'used',
'states',
'means',
'put',
'schools',
'5']

for(let i=0;i < stop_words.length;i++)
{
  stop_words[i]+= ','
  
  stop_words[i]+= " "
}
function App() {

  const [bayesResult,setBayesResult] = useState("")
  const [bayesTokenize,setBayesTokenize] = useState([""])
  const [bertResult,setBertResult] = useState("")
  const [bertTokenize,setBertTokenize] = useState([""])


  const passBayesResult = (a) =>{
    setBayesResult(a) 
  }

  const passBayesTokenize = (a) =>{
    setBayesTokenize(a)
  }

  const passBertResult = (a) =>{
    setBertResult(a)
  }

  const passBertTokenize = (a) =>{
    setBertTokenize(a)
  }

  console.log(myData)



  return (
    <div className="App" style = {{backgroundColor:'#FBFBF9'}}>


    
    <div style = {{backgroundColor:"#F3EAD3", margin:'5px 30px 0 30px', borderRadius:'10px', textAlign:'center' }}>
      <div style = {{textAlign:'center',paddingTop:'2px'}}>
        <h2>Plots</h2>
      </div>

<div style ={{margin:'10px 20px 0px 20px',display:'flex', borderRadius:'10px',backgroundColor:'#F3EAD3'}}>
    <div style={{width:'50%',margin:'auto'}}>
      <BarChartForClass onGetType = {() => {return "normal"}} onGetName = {() => {return "Classes"} } />
    </div>

    <div style = {{width:'50%',margin:'auto'}}>
      <BarChartForClass onGetType = {() => {return "balance"}} onGetName = {() => {return "Balanced classes"}} />

    </div>

</div>

</div>

<div style = {{margin: '20px 30px 10px 30px' ,borderRadius:'10px',backgroundColor:'#F3EAD3'}}>
<div style = {{textAlign : 'center', paddingTop:'5px'}}>
</div>
   
  <BarChartForStopWords />

</div>

    <div style = {{display:'flex', margin:'20px 30px 0 30px',marginTop:'20px',borderRadius:'10px',backgroundColor:'#F3EAD3' }}>

      <div style ={{width:'50%'}}>
        <h3>Bayes Model</h3>
        <InputBar onPassBayesResult = {passBayesResult} onPassTokenize = {passBayesTokenize} />
        
        {bayesTokenize.length > 0?(<div>
          <div>
            <h2>Tokenization</h2>
            </div>
        <div style = {{display: 'flex',marginTop:'20px',marginLeft:'5px',margin:'auto' }}>

          {bayesTokenize.map(
            (e) => <div style = {{backgroundColor:'#88CFF1', marginRight:'10px',borderRadius:'5px',padding:'2px' }}>
                {e}
            </div> 
          )}
        </div> 
        </div>):(<div></div>)}

      {bayesResult !== "" ?( 
        <div>
          <h2>Result : {bayesResult}</h2>
        </div>):(<div></div>)}

      </div>

      <div style ={{width:'50%'}}>
        <h3>BERT model</h3>
          <InputBarForBert onPassBertResult = {passBertResult} onPassBertTokenize = {passBertTokenize}/>
            
          {bertTokenize.length > 0?(<div>
          <div>
            <h2>Tokenization</h2>
            </div>
        <div style = {{display: 'flex',marginTop:'20px',marginLeft:'5px',margin:'auto' }}>

          {bertTokenize.map(
            (e) => <div style = {{backgroundColor:'#88CFF1', marginRight:'10px',borderRadius:'5px',padding:'2px' }}>
                {e}
            </div> 
          )}
        </div> 
        </div>):(<div></div>)}

      {bertResult !== "" ? (
        <div>
          <h2>Result : {bertResult}</h2>
        </div>):(<div></div>)}

      </div>
    </div>

    </div>
  );
}

export default App;
