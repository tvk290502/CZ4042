import jsons



from flask import Flask, send_file, make_response
from flask import Flask, request, render_template
app = Flask(__name__)


# dataset = load_dataset('csv', data_files={'train': base_url+'bert_train.csv','validation': base_url+'bert_val.csv','test': base_url+'bert_test.csv'})


@app.route('/test', methods = ['GET', 'POST','DELETE', 'PATCH'])
def main():
   length = ""
   if request.method == 'POST':
        length = request.get_json()['input']
   print(length)
   
   d = {
     "check":1,
     "result":"bingsu" }
   
   return d


  
if __name__ == '__main__':
    app.run(host="localhost",port=5000)
