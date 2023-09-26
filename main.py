from flask import Flask, render_template, flash, request, redirect, url_for
from werkzeug.utils import secure_filename

from utils.utils import get_user_uploads_paths, allowed_file, clean_up
import os
 
app = Flask(__name__)
app.secret_key = "solvento-omdena"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
@app.route('/')
def home():
     #return render_template('index.html', output_list=[], pred_img_paths=[], no_detect_imgs=[], sign_stamp_count=[], times={})
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the files part
        if 'files[]' not in request.files:
            flash('No file part')
            return redirect(request.url)

        files = request.files.getlist('files[]')
        pdf_root_dir =  get_user_uploads_paths()

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(pdf_root_dir, filename))

        flash('File(s) successfully uploaded')

        # Get user input value from front-end
        input_invoice_ids = request.form['user-input-invoice-ids']
        input_invoice_ids = input_invoice_ids.split(',')
        input_ocr_model = request.form.get('input-ocr-model')
        
        # AI Pipeline
        # no_detection_images, sign_stamp_count, ocr_result_list, \
        #     times, current_pred_paths = process_ai_pipeline(input_invoice_ids=input_invoice_ids, input_ocr_model=input_ocr_model)
        # clean_up()
        # return render_template('index.html', 
        #                         output_list=ocr_result_list, 
        #                         filenames=current_pred_paths,
        #                         no_detect_imgs=no_detection_images,
        #                         sign_stamp_count=sign_stamp_count,
        #                         times=times)
        return render_template('index.html')

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)
    
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)       