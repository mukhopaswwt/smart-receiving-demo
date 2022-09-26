from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
import os, os.path
import cv2
import numpy as np
import pandas as pd
import re
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import pytesseract

pytesseract.pytesseract.tesseract_cmd = os.getenv('TESSERACT_PATH')

po_num='test'

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(app.root_path, 'data', 'images')
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def fuzzy_matching(df, po_num):
    '''
    Fuction to check if the content from the OCR is matching with WWT database file
    :param df: table dataframe extracted using OCR
    :return: final dataframe with matching content
    '''
    po_string = "po_" + str(po_num)+ '.csv'
    database_filename = os.path.join(os.path.abspath(os.getcwd()), 'WWTPO_datasets', po_string)
    

    #database_filename = '.\WWTPO_datasets\po_' + str(po_num) + '.csv'
   
    dell_po=pd.read_csv(os.path.join(app.root_path, database_filename), header=1)
    dell_po=dell_po[['WWT PO Number','Serial Number','Quantity','Item Description']].drop_duplicates()
    print('dell_df shape', dell_po.shape)
    df = df.add_prefix('ocr_')
    print('df_shape', df.shape)
    dell_po1=pd.merge(dell_po, df, left_on=['Serial Number'], right_on=['ocr_serial_number'], how='left')
    ###Mismatch data###
    dell_po1_nf=dell_po1[dell_po1['ocr_serial_number'].isna()]

    ###Matched data###
    dell_po1_f=dell_po1[dell_po1['ocr_serial_number'].notna()]
    
    ###Matching Quantity###
    dell_po1_f["ocr_quantity"] = dell_po1_f["ocr_quantity"].astype(str).astype(int)
    dell_po1_f['Match'] = np.where(dell_po1_f['Quantity'] == dell_po1_f['ocr_quantity'], 'Matched','Not Matched')
    dell_po1_nf = dell_po1_nf[['WWT PO Number','Serial Number','Quantity','Item Description']]

    mat1 = []
    mat2 = []
    p = []

    list1 = dell_po1_nf['Serial Number'].tolist()
    list2 = df[~df['ocr_serial_number'].isin(dell_po1_f['ocr_serial_number'].tolist())]['ocr_serial_number'].tolist()
    threshold = 80

    # iterating through list1 to extract
    # it's closest match from list2
    for i in list1:
        mat1.append(process.extractOne(
          i, list2, scorer=fuzz.token_sort_ratio))
    dell_po1_nf['Fuzzy Match Text'] = mat1
    # iterating through the closest matches
    # to filter out the maximum closest match
    for j in dell_po1_nf['Fuzzy Match Text']:
        print(j)
        print(type(j))
        if j[1] >= threshold:
            p.append(j[0])
        mat2.append(",".join(p))
        p = []

    # storing the resultant matches back
    # to dframe1
    dell_po1_nf['Fuzzy Match Text'] = mat2
    dell_po1_fuzzy_nf=dell_po1_nf[dell_po1_nf['Fuzzy Match Text']=='']
    dell_po1_fuzzy_nf['Match']='Not Matched'
    dell_po1_nf=dell_po1_nf[dell_po1_nf['Fuzzy Match Text']!='']
    
    dell_po1_nf=pd.merge(dell_po1_nf, df,left_on=['Fuzzy Match Text'], right_on=['ocr_serial_number'], how='left')
    dell_po1_nf["ocr_quantity"] = dell_po1_nf["ocr_quantity"].astype(str).astype(int)
    dell_po1_nf['Match'] = np.where(dell_po1_nf['Quantity'] == dell_po1_nf['ocr_quantity'], 'Matched','Not Matched')

    final_df=pd.concat([dell_po1_f,dell_po1_nf,dell_po1_fuzzy_nf])
    final_df.rename(columns={'ocr_serial_number': 'OCR Serial Number', 'ocr_quantity': 'OCR Quantity'}, inplace=True)
    final_df["Fuzzy Match Text"].fillna('-', inplace=True)
    return final_df

def image_read(path):
    img = cv2.imread(path)
    return img

def preprocessing(img, vendor):
    '''
    Function to preprocess the given image according to their vendor name
    :param img: image in vector form
    :param vendor: vendor name for the given image
    :return: preprocessed image vector
    '''
    # Rescale the image, if needed.
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    # Convert to gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply dilation and erosion to remove some noise
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    # Apply blur to smooth out the edges
    img = cv2.GaussianBlur(img, (5, 5), 0)
    # Thresholding
    if vendor=='CISCO':
        (thresh, img) = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY)
        
    if vendor=='HP':
        (thresh, img) = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY)
        
    if vendor=='DELL':
        (thresh, img) = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
        
    if vendor=='Arista':
        (thresh, img) = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY)

    return img

def image_crop_table(img, vendor):
    '''
    Function to crop the table out of a given image
    :param img: image in vector form
    :param vendor: vendor name for the given image
    :return: cropped image vector
    '''
    # Writing the img matrix into a file
    cv2.imwrite(os.path.join(app.root_path, 'data\images\temp.png'), img)
    
    height, width = img.shape
    # Cropping image based on hard coded values for every vendor
    if vendor=='CISCO':
        img1= img[int(height*.50):height, 0:width]
        return img1

    if vendor=='HP':
        img1= img[int(height/2)-int(height*.1):height, 0:width]
        cv2.imwrite(os.path.join(app.root_path, 'ac.png'), img1)
        return img1

    if vendor=='DELL':
        img1 = img[int(height * .3):int(height * .8), 0:width]
        cv2.imwrite(os.path.join(app.root_path, 'ac.png'), img1)
        return img1

    if vendor=='Arista':
        img1= img[int(height*.5):int(height*.7), 0:width]
        return img1

def image_crop_po(img, vendor):
    '''
    Function to crop the PO number out of a given image
    :param img: image in vector form
    :param vendor: vendor name for the given image
    :return: cropped image vector
    '''
    height, width = img.shape

    # Cropping image based on hard coded values for every vendor
    if vendor=='CISCO':
        img1 = img[0:int(height * .2), int(width * .7):width]
        return img1
    if vendor=='HP':
        img1= img[0:int(height*.5), int(width*.60):width]
        return img1
    if vendor=='DELL':
        img1= img[0:int(height*.3), int(width*.2):int(width*.5)]
        return img1
    if vendor=='Arista':
        img1= img[int(height*.3):int(height*.45), 0:int(width*.5)]
        return img1

def extract_po(extracted_string, vendor):
    '''
    Function to obtain the PO number from the string extracted by tesseract
    :param extracted_string: string extracted by tesseract
    :param vendor: vendor name for the given extracted image
    :return: PO number
    '''
    if vendor=='CISCO':
        lines = extracted_string.split('\n')
        ls1 = []
        pattern = '\S+\d{7,100}|\d{7,100}'

        for line in lines:
            matchObj = re.search(pattern, line)
            if matchObj:
                ls1.append(line)
        pattern='\d{1,100}$'
        matchObj = re.search(pattern, ls1[0])
        matchObj = matchObj[0]
        return matchObj

    if vendor=='HP':
        lines = extracted_string.split('\n')
        ls1=[]
        pattern='Customer PO'
        for line in lines:
            matchObj = re.search(pattern, line)
            if matchObj:
                ls1.append(line)
        pattern='\d{1,100}$'
        matchObj = re.search(pattern, ls1[0])
        matchObj = matchObj[0]
        return matchObj
    
    if vendor=='DELL':
        pattern = 'PO[.|\n|\W|\w]*'
        matchObj = re.search(pattern, extracted_string)
        pattern = '\d{6,20}'
        matchObj = re.search(pattern, extracted_string)
        matchObj = matchObj[0]
        return matchObj
    
    if vendor=='Arista':
        lines = extracted_string.split('\n')
        
        ls1=[]
        pattern='PO'
        for line in lines:
            matchObj = re.search(pattern, line)
            if matchObj:
                ls1.append(line)
        
        pattern='\d{1,100}$'
        matchObj = re.search(pattern, ls1[0])
        matchObj = matchObj[0]

def extract_table(extracted_string, vendor, vendor_specific='none'):
    '''
    Function to obtain the content of the table from the string extracted by tesseract
    :param extracted_string: string extracted by tesseract
    :param vendor: vendor name for the given extracted image
    :return: content of the table as a data frame
    '''
    if vendor=='CISCO':
        if vendor_specific=='none':
        ###get the lines where we have a pattern of d d ex 1 1 or 2 2
            lines = extracted_string.split('\n')
            ls1=[]
            pattern='\d+ \d+'
            for line in lines:
                matchObj = re.search(r"\d+ \d+", line)
                if matchObj:
                    ls1.append(line)

            ###extract quantity###
            ls2=[]
            for item in ls1:
                try:
                    res = re.search('\d+ \d+$', item).group()
                    ls2.append(res)
                except:
                    res = re.search('\d+ \d+', item).group()
                    ls2.append(res)

            ###extract part num####
            part_num = [re.sub(r'^[\d]+\s', '', text).strip() for text in ls1]
            part_num = [text.split(" ")[0] for text in part_num]

            ###getting item desc####
            item_desc = [re.sub(r'\d+ \d+.*$', '', text).strip() for text in ls1]

            ###creating dataframe####
            df = pd.DataFrame(list(zip(part_num, ls2, item_desc)),columns =['part_num', 'quantity','item_desc'])

            ###cleaning item desc####
            df['item_desc']= df.apply(lambda x: x['item_desc'].replace(x['part_num'], ''), axis=1)
            df['item_desc']=df['item_desc'].replace(to_replace=r'^[\d]+\s',value ='', regex=True)

            ###split quantity column to order and ship quantity####
            new = df["quantity"].str.split(" ", n = 1, expand = True)
            df['order_quantity']= new[0]
            df['ship_quantity'] = new[1]
            df.drop(columns =["quantity","order_quantity","item_desc"], inplace = True)
            df=df[['ship_quantity','part_num']]
            return df
        if vendor_specific == 'carton':
            print(extracted_string)
            lines = extracted_string.split('\n')
            lines = [x.strip(' ') for x in lines]
            lines = [i for i in lines if i]
            lines = extracted_string.split('\n')
            # print(lines)
            ls1 = []
            for line in lines:
                matchObj = re.search(r"CARTONID|CARTON ID", line)
                if matchObj:
                    ls1.append(line)
            print(ls1)
            ls1 = [re.sub(r"[^a-zA-Z0-9 ]", "", x) for x in ls1]
            ls1 = [re.sub("  ", " ", x) for x in ls1]
            ls1 = [re.sub("CARTON ID", "CARTONID", x) for x in ls1]
            ls1 = [re.sub("SERIAL NO", "SERIALNO", x) for x in ls1]
            ls2 = []
            print(ls1)
            for line in ls1:
                matchObj = re.search(r"SERIALNO \w+", line).group()
                if matchObj:
                    ls2.append(matchObj)
            df = pd.DataFrame(list(zip(ls2)), columns=['raw_line'])
            new = df["raw_line"].str.split(" ", n=1, expand=True)
            df['product'] = new[1]
            df['quantity'] = 1
            df = df[['quantity', 'product']]
            return df


    if vendor=='HP':
        ###get the lines where we have a pattern of d d ex 1 1 or 2 2
        lines = extracted_string.split('\n')
        ls1=[]
        pattern='^\S* \d+ |^\S+ \S+ \d+ '
        for line in lines:

            matchObj = re.search(pattern, line)
            if matchObj:
                ls1.append(line)
        ###creating dataframe####
        df = pd.DataFrame(list(zip(ls1)),columns =['raw_line'])
        
        ###cleaning item desc####
        df['placeholder']=df['raw_line'].str.extract('(^\S* \d+ |^\S+ \S+ \d+ )', expand=True)
        df['product']=df['placeholder'].str.extract('^(.*?)\d+ $', expand=True)
        df['quantity']=df['placeholder'].str.extract('(\d+ $)', expand=True)
        df['item_desc']= df.apply(lambda x: x['raw_line'].replace(x['placeholder'], ''), axis=1)

        df.columns = df.columns.str.strip()
        df['product']=df['product'].replace(to_replace=r'^\w+ ',value ='', regex=True)
        df.columns = df.columns.str.strip()
        df.drop(columns =["raw_line","placeholder"], inplace = True)
        df=df[['quantity','product']]
        return df
    
    if vendor=='DELL':
        ###get the lines where we have a pattern of d d ex 1 1 or 2 2
        lines = extracted_string.split('\n')
        #print(lines)
        ls1=[]
        pattern='^\d+ \d+ '
        for line in lines:
            matchObj = re.search(pattern, line)
            if matchObj:
                ls1.append(line)
                
        ###creating dataframe####
        df = pd.DataFrame(list(zip(ls1)),columns =['raw_line'])
        df['quantity']=df['raw_line'].str.extract('^\d+ (\d+ )', expand=True)
        df['serial_number']=df['raw_line'].str.extract('(\w+)$', expand=True)
        df=df[['quantity','serial_number']]
        return df
 
def tesserect_engine(img1, vendor_specific='none'):
    '''
    Function to extract string from a given image
    :param img1:
    :param vendor_specific:
    :return:
    '''

    if vendor_specific == 'carton':
        extracted_string = pytesseract.image_to_string(img1, lang="eng", config='--psm 4')
    else:
        extracted_string = pytesseract.image_to_string(img1, lang="eng")
    return extracted_string

@app.route('/')
def home_main():
    return render_template('home.html')

@app.route('/index', methods = ['GET', 'POST'])
def index_main():
    # Upload file in the desired path
    if request.method == 'POST':
        global vendor
        global img_loc
        vendor = request.form['vendor']

        f = request.files['files']

        f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
        img_loc =  os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
    
    return render_template('index.html')


@app.route('/poinfo', methods = ['GET', 'POST'])
def po_data():
    if request.method == 'POST':
        global vendor
        global img_loc
        vendor = request.form['vendor']

        f = request.files['files']

        f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
        img_loc =  os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
        img=image_read(img_loc)
        
        
        img=preprocessing(img,vendor)
        img1=image_crop_po(img, vendor)
        extracted_string=tesserect_engine(img1)
        try:
            extracted_po=extract_po(extracted_string, vendor)
            return render_template('ponum.html', po_num=extracted_po, vendor=vendor, img_loc=img_loc)
        except:
            return render_template("error.html")


@app.route('/match', methods = ['GET', 'POST'])
def match_text():

    f = request.form.to_dict(flat=False)
    #print(f)
    df = pd.DataFrame.from_dict(f)
    x = fuzzy_matching(df, po_num)
    txt='final'
    return render_template('final_output.html',  tables=[x.to_html(classes=['table', 'table-striped', 'table-bordered', 'table-condensed'], header="true", index=False)],po_num=po_num, final=txt)



@app.route('/asn', methods = ['GET', 'POST'])
def asn_check():
    if request.method == 'POST':
        global po_num
        po_num = request.form['po_num']
        global vendor
        vendor = request.form['vendor']
        global img_loc
        img_loc = request.form['img_loc']

        
        # df1=pd.read_csv(os.path.join(app.root_path, 'data', 'intermediate_data', 'asn_mock_database.csv'))

        # try:
        #     df1=df1[df1['PO Number']==int(po_num)]
        # except:
        #     df1 = df1[df1['PO Number'] == po_num]

        # if df1.shape[0]>0 :
        if False:
            return render_template('final_output.html',  tables=[df1.to_html(classes=['table', 'table-striped', 'table-bordered', 'table-condensed'], header="true", index=False)],po_num=po_num)

        else:
            print(po_num)
            print(vendor)
            print(img_loc)
            img=image_read(img_loc)
            print('image_read done')
            img=preprocessing(img,vendor)
            print('image_preprocessing done')
            img1=image_crop_table(img, vendor)
            print('image_crop done')
            extracted_string=tesserect_engine(img1)
            print('tesseract done')

            if vendor=='CISCO':

                matchObj = re.search(r"CARTONID|CARTON ID", extracted_string)
                if matchObj is None:
                    extracted_table = extract_table(extracted_string, vendor)
                else:
                    extracted_string = tesserect_engine(img1,vendor_specific='carton')
                    extracted_table = extract_table(extracted_string, vendor, vendor_specific='carton')

            else:
                extracted_table = extract_table(extracted_string, vendor)

            print('table done')
            extracted_table.to_csv(os.path.join(app.root_path, 'data', 'intermediate_data', 'final.csv'), index=False)

            df1=pd.read_csv(os.path.join(app.root_path, 'data', 'intermediate_data', 'final.csv'))
            df1['id'] = range(0, len(df1))
            df1.columns = ['quantity', 'serial', 'id']
            df_Final_header = list(df1)
            print(df1)
            return render_template('table.html', table = df1.values, headers=df_Final_header, po_num=po_num)
            
@app.route('/trynow', methods = ['GET', 'POST'])
def try_now():
    global po_num
    po_num = '3902748'
    global vendor
    vendor = 'CISCO'
    global img_loc
    img_loc = '.\data\images\CISCO1_0_7.png'

    print(po_num)
    print(vendor)
    print(img_loc)
    img=image_read(img_loc)
    print('image_read done')
    img=preprocessing(img,vendor)
    print('image_preprocessing done')
    img1=image_crop_table(img, vendor)
    print('image_crop done')
    extracted_string=tesserect_engine(img1)
    print('tesseract done')

    if vendor=='CISCO':

        matchObj = re.search(r"CARTONID|CARTON ID", extracted_string)
        if matchObj is None:
            extracted_table = extract_table(extracted_string, vendor)
        else:
            extracted_string = tesserect_engine(img1,vendor_specific='carton')
            extracted_table = extract_table(extracted_string, vendor, vendor_specific='carton')

    else:
        extracted_table = extract_table(extracted_string, vendor)

    print('table done')
    extracted_table.to_csv(os.path.join(app.root_path, 'data', 'intermediate_data', 'final.csv'), index=False)

    df1=pd.read_csv(os.path.join(app.root_path, 'data', 'intermediate_data', 'final.csv'))
    df1['id'] = range(0, len(df1))
    df1.columns = ['quantity', 'serial', 'id']
    df_Final_header = list(df1)
    print(df1)
    return render_template('trnow.html', table = df1.values, headers=df_Final_header, po_num=po_num)

@app.route('/demo')
def demo():
    return render_template("image.html")

@app.route('/download')
def download_file():
    p= "sample_images.zip"
    return send_file(p, as_attachment=True)

if __name__ == '__main__':
    app.run(host= '0.0.0.0', debug = True)

