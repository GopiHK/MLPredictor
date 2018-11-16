from flask import render_template, request, flash, redirect, Blueprint, url_for
from werkzeug import secure_filename
import os, re
from flask import current_app
from spamfilter.models import db, File
import json
from collections import OrderedDict
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

from spamfilter.forms import InputForm
from spamfilter import spamclassifier
from instance.config import ALLOWED_EXTENSIONS



spam_api = Blueprint('SpamAPI', __name__)

def allowed_file(filename, extensions=None):
    '''
    'extensions' is either None or a list of file extensions.
    
    If a list is passed as 'extensions' argument, check if 'filename' contains
    one of the extension provided in the list and return True or False respectively.
    
    If no list is passed to 'extensions' argument, then check if 'filename' contains
    one of the extension provided in list '', defined in 'config.py',
    and return True or False respectively.
    '''
    extension = os.path.splitext(filename)[1][1:]
        
    if  extensions is not None:
        if extension in extensions :
            return True
        else:
            return False
    else:
        if extension in  ALLOWED_EXTENSIONS:
            return True
        else:
            return False

        
        

@spam_api.route('/')
def index():
    '''
    Renders 'index.html'
    '''
    return render_template('index.html')

@spam_api.route('/listfiles/<success_file>/')
@spam_api.route('/listfiles/')
def display_files(success_file=None):
    '''
    Obtain the filenames of all CSV files present in 'inputdata' folder and
    pass it to template variable 'files'.
    
    Renders 'filelist.html' template with values  of varaibles 'files' and 'fname'.
    'fname' is set to value of 'success_file' argument.
    
    if 'success_file' value is passed, corresponding file is highlighted.
    '''
    files=[ filename for filename  in os.listdir('./inputdata') if  os.path.splitext(filename)[1][1:]=="csv"  ]
    return render_template('filelist.html' ,files=files, fname=success_file )
        
        
    




def validate_input_dataset(input_dataset_path):
    '''
    Validate the following details of an Uploaded CSV file
    
    1. The CSV file must contain only 2 columns. If not display the below error message.
    'Only 2 columns allowed: Your input csv file has '+<No_of_Columns_found>+ ' number of columns.'
    
    2. The column names must be "text" nad "spam" only. If not display the below error message.
    'Differnt Column Names: Only column names "text" and "spam" are allowed.'
    
    3. The 'spam' column must conatin only integers. If not display the below error message.
    'Values of spam column are not of integer type.'
    
    4. The values of 'spam' must be either 0 or 1. If not display the below error message.
    'Only 1 and 0 values are allowed in spam column: Unwanted values ' + <Unwanted values joined by comma> + ' appear in spam column'
    
    5. The 'text' column must contain string values. If not display the below error message.
    'Values of text column are not of string type.'
    
    6. Every input email must start with 'Subject:' pattern. If not display the below error message.
    'Some of the input emails does not start with keyword "Subject:".'
    
    Return False if any of the above 6 validations fail.
    
    Return True if all 6 validations pass.
    
    '''
    
    
    data=pd.read_csv(input_dataset_path)

    col=data.columns


    No_of_Columns_found=len(col)


    if No_of_Columns_found!=2:
        flash('Your input csv file has '+str(No_of_Columns_found)+ ' number of columns.')
        return False



    if   col[0]!="text" or  col[1]!="spam":
        flash('Only column names "text" and "spam" are allowed.')
        return False
    no_unwanted_val=[ i for i in data['spam' ].unique() if i not in [1,0] ]

    non_Texts=[t  for t in data["text"]  if isinstance(t,str)==False]

    invalid_text=[t  for t in data["text"]  if t.startswith('Subject:')==False]



    if data['spam' ].unique().dtype not in ['int64','int32']:
        flash('Values of spam column are not of integer type.')
        return False
    if len(no_unwanted_val)>0 :
        flash('Unwanted values ' + ",".join([str(x) for x in no_unwanted_val])   + ' appear in spam column')
        return False
    if len(non_Texts)>0 :
        flash('Values of text column are not of string type.')
        return False
    if len(invalid_text)>0 :
        flash('Some of the input emails does not start with keyword "Subject:".')
        return False
    return True

@spam_api.route('/upload/', methods=['GET', 'POST'])
def file_upload():
    if request.method == 'GET':
        return render_template('upload.html')
    else:
        f=request.files['uploadfile']
        if allowed_file(f.filename)== False:
            flash('Only CSV Files are allowed as Input.')
            return render_template('upload.html')

        else:

            filename = secure_filename(f.filename)
            path=os.path.join(os.getcwd(),'inputdata',filename)
            df=pd.read_csv(f)
            df.to_csv(path,index=False)
            if validate_input_dataset(path)==False:
                os.remove(path)
                return render_template('upload.html')
            else:
                db.session.add(File(name=filename,filepath=os.path.abspath(path)))
                db.session.commit()
                return redirect(url_for('.display_files',success_file=filename)) 


def validate_input_text(intext):
    
   
    
    od1 = OrderedDict()
    k=0
    st,en=0,0# Start and End points 
    a=intext#Input text
    for i,line in enumerate(a):#looping text with enumerate index
        if  a[ i:i+8].strip()=="Subject:" and i>0 :# checking is der any string having Subject:
            #if there
            en=i
            t=a[st:en]#Extract the Sub string from st to en
            n=t[:30]# first 30 charcter of extracted string
            m=t.strip()#stripping the spaces
            
           
            if a[en-4:en+4].strip().count('\n')<2 : #here checking is there any blank line 
                #no blank line 
                return False
            
            #yes then store
            
            od1[str(k)+"loop"]=m
            k+=1
            st=i
            
    
    if a[en-4:en+4].strip().count('\n')<2 and en >0:# end date is set to where last subject starts
        return False
    else:
        if a[en:8].strip()=="Subject:" and len(od1)==0 and en==0: 
            
            t=a[st:len(a)]#recieving the last srtring
            n=t[:30]
            m=t.strip()
            od1[str(k)+"No"]=m
        elif a[en:8].strip()!="Subject:" and len(od1)==0 and en==0:
            return False
        elif len(od1)>0:
            t=a[st:len(a)]#recieving the last srtring
            n=t[:30]
            m=t.strip()
            od1[str(k)+"last"]=m
    return od1 
        
        


@spam_api.route('/models/<success_model>/')
@spam_api.route('/models/')
def display_models(success_model=None):
    
    '''
    Obtain the filenames of all machine learning models present in 'mlmodels' folder and
    pass it to template variable 'files'.
    
    NOTE: These models are generated from uploaded CSV files, present in 'inputdata'.
    So if ur csv file names is 'sample.csv', then when you generate model
    two files 'sample.pk' and 'sample_word_features.pk' will be generated.
    
    Consider only the model and not the word_features.pk files.
    
    Renders 'modelslist.html' template with values  of varaibles 'files' and 'model_name'.
    'model_name' is set to value of 'success_model' argument.
    
    if 'success_model value is passed, corresponding model file name is highlighted.
    '''
    files=[ filename for filename  in os.listdir('./mlmodels')   ]
    return render_template('modelslist.html' ,files=files, model_name=success_model )



def isFloat(value):
    '''
    Return True if <value> is a float, else return False
    '''
    try:
        V_value=float(value)
        return True
    except ValueError:
        return False
        
    
    


def isInt(value):
    '''
    Return True if <value> is an integer, else return False
    '''
    try:
        V_value=int(value)
        return True
    except ValueError:
        return False

@spam_api.route('/train/', methods=['GET', 'POST'])
def train_dataset():
    
    '''
    If request is of GET method, render 'train.html' template with tempalte variable 'train_files',
    set to list if csv files present in 'inputdata' folder.
    
    If request is of POST method, capture values associated with
    'train_file', 'train_size', 'random_state', and 'shuffle'
    
    if no 'train_file' is selected, render the same page with GET Request and below error message.
    'No CSV file is selected'
    
    
    if 'train_size' value is not float, render the same page with GET Request and below error message.
    'Training Data Set Size must be a float.
    
    if 'train_size' value is not in between 0.0 and 1.0, render the same page with GET Request and below error message.
    'Training Data Set Size Value must be in between 0.0 and 1.0'
    
    if 'random_state' is None,render the same page with GET Request and below error message.
    'No value
 provided for random state.''
    
    if 'random_state' value is not an integer, render the same page with GET Request and below error message.
    'Random State must be an integer.'
    
    if 'shuffle' is None, render the same page with GET Request and below error message.
    'No option for shuffle is selected.'
    
    if 'shuffle' is set to 'No' when 'Startify' is set to 'Yes', render the same page with GET Request and below error message.
    'When Shuffle is No, Startify cannot be Yes.'
    
    If all input values are valid, build the model using submitted paramters and methods defined in
    'spamclassifier.py' and save the model and model word features file in 'mlmodels' folder.
    
    NOTE: These models are generated from uploaded CSV files, present in 'inputdata'.
    So if ur csv file names is 'sample.csv', then when you generate model
    two files 'sample.pk' and 'sample_word_features.pk' will be generated.
    
    Finally render, 'display_models' template with value of template varaible 'success_model'
    set to name of model generated, ie. 'sample.pk'
    '''
    if request.method == 'GET':
        files=[ filename for filename  in os.listdir('./inputdata') if  os.path.splitext(filename)[1][1:]=="csv"  ]
        return render_template('train.html',train_files=files)
    else:
        train_file=request.form.get("train_file")
       
        if train_file is None :
            flash('No CSV file is selected')
            return redirect(url_for('.train_dataset'))
        v_train_size = request.form.get('train_size')
        if isFloat(v_train_size)==False:
            flash('Training Data Set Size must be a float')
            return redirect(url_for('.train_dataset'))
        v_train_size=float(v_train_size)
        
        if v_train_size<0.0 or v_train_size>1.0:
            flash('Training Data Set Size Value must be in between 0.0 and 1.0')
            return redirect(url_for('.train_dataset'))
        V_random_state = request.form.get( 'random_state')
        if V_random_state is None:
            flash('No value provided for random state.')
            return redirect(url_for('.train_dataset'))
        if isInt(V_random_state)==False:
            flash('Random State must be an integer.')
            return redirect(url_for('.train_dataset'))
        
        V_random_state=int(V_random_state)
        V_shuffle = request.form.get('shuffle')
        V_stratify = request.form.get('stratify')
        if V_shuffle is None:
            flash('No option for shuffle is selected.')
            return redirect(url_for('.train_dataset'))
        if V_stratify == 'Y' and V_shuffle=='N':
            flash('When Shuffle is No, Startify cannot be Yes.')
            return redirect(url_for('.train_dataset'))
        classifier = spamclassifier.SpamClassifier()
        
        path=os.path.join(os.getcwd(),'inputdata','sample.csv')
        
        data =pd.read_csv(path)
        train_X, test_X, train_Y, test_Y = train_test_split(data["text"].values,
                                                            data["spam"].values,
                                                            test_size = 1- 0.80,
                                                            random_state = 42,
                                                            shuffle = 'N',
                                                            stratify=data["spam"].values)
                                                            
        classifier_model, model_word_features = classifier.train(train_X, train_Y)
        os.path.splitext(train_file)[0]
        model_name = os.path.join(os.getcwd(),'mlmodels',os.path.splitext(train_file)[0]+'.pk')
        model_word_features_name =os.path.join(os.getcwd(),'mlmodels',os.path.splitext(train_file)[0]+'_word_features.pk')
        with open(model_name, 'wb') as model_fp:
            pickle.dump(classifier_model, model_fp)
        with open(model_word_features_name, 'wb') as model_fp:
            pickle.dump(model_word_features, model_fp)
        
        
    
    
    return  redirect(url_for('.display_models',success_model=model_name))
    

@spam_api.route('/results/')
def display_results():
    '''
    Read the contents of 'predictions.json' and pass those values to 'predictions' template varaible
    
    Render 'displayresults.html' with value of 'predictions' template variable.
    '''
    with open('predictions.json', 'r') as f:
        distros_dict = json.load(f)

    return render_template('displayresults.html',predictions=distros_dict)
    
@spam_api.route('/predict/', methods=['GET', "POST"])
def predict():
    '''
    If request is of GET method, render 'emailsubmit.html' template with value of template
    variable 'form' set to instance of 'InputForm'(defined in 'forms.py').
    Set the 'inputmodel' choices to names of models (in 'mlmodels' folder), with out extension i.e .pk
    
    If request is of POST method, perform the below checks
    
    1. If input emails is not provided either in text area or as a '.txt' file, render the same page with GET Request and below error message.
    'No Input: Provide a Single or Multiple Emails as Input.'
    
    2. If input is provided both in text area and as a file, render the same page with GET Request and below error message.
    'Two Inputs Provided: Provide Only One Input.'
    
    3. In case if input is provided as a '.txt' file, save the uploaded file into 'inputdata' folder and read the
     contents of file into a variable 'input_txt'
    
    4. If input provided in text area, capture the contents in the same variable 'input_txt'.
    
    5. validate 'input_txt', using 'validate_input_text' function defined above.
    
    6. If 'validate_input_text' returns False, render the same page with GET Request and below error message.
    'Unexpected Format : Input Text is not in Specified Format.'

    
    7. If 'validate_input_text' returns a Ordered dictionary, choose a model and perform prediction of each input email using 'predict' method defined in 'spamclassifier.py'
    
    8. If no input model is choosen, render the same page with GET Request and below error message.
    'Please Choose a single Model'
    
    9. Convert the ordered dictionary of predictions, with 0 and 1 values, to another ordered dictionary with values 'NOT SPAM' and 'SPAM' respectively.
    
    10. Save thus obtained predictions ordered dictionary into 'predictions.json' file.
    
    11. Render the template 'display_results'
    
    '''
    form=InputForm()
    form.inputmodel.choices=[ (os.path.splitext(filename)[0],os.path.splitext(filename)[0]) for filename  in os.listdir('./mlmodels') if  filename.endswith('word_features.pk')==False  ]

    if request.method == 'GET':
        
        return render_template('emailsubmit.html',form=form)
    else:
        inputmail=request.form.get('inputemail') 
        inputfile=request.files.get('inputfile')
        inputmodel=request.form.get('inputmodel')
        if len(inputmail)==0 and inputfile is None:
            flash('No Input: Provide a Single or Multiple Emails as Input.')
            return render_template('emailsubmit.html',form=form)
        
        if len(inputmail)>0 and inputfile is not None:
            flash('Two Inputs Provided: Provide Only One Input.')
            return render_template('emailsubmit.html',form=form)
        
        if inputfile is not None:
            path=os.path.join(os.getcwd(),'inputdata', inputfile.filename)
            with open(path,'wb') as f:
                f.write (inputfile.read())
            
            with open (path, "r") as myfile:
                input_txt=myfile.read()
                
        else:
            input_txt=inputmail
            
        
        
        temp=validate_input_text(input_txt)
      
            
        if temp==False:
            flash( 'Unexpected Format : Input Text is not in Specified Format.')
            return render_template('emailsubmit.html',form=form)
        
        if inputmodel is None:
            flash('Please Choose a single Model')
            return render_template('emailsubmit.html',form=form)
        
        ordered_dic=OrderedDict()
        
        classifier = spamclassifier.SpamClassifier()
        
        classifier.load_model(inputmodel)
        
        
        for  key, value in temp.items():
            t=classifier.predict(value)
            if t[0]== 0:
                ordered_dic[value]='NOT SPAM'
               
            else:
                ordered_dic[value]='SPAM'
        
                
        with open('predictions.json', 'w') as f:
            json.dump(ordered_dic, f)
            
        return redirect(url_for('.display_results'))
    