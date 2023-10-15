from flask import Flask, render_template, flash, request, redirect, url_for
from werkzeug.utils import secure_filename

from utils.utils import get_user_uploads_paths, clean_up, seed_everything, allowed_file
from predict import Yolov8
import os
from glob import glob
import secrets
import shutil
 
app = Flask(__name__)
seed_everything()
app.secret_key = secrets.token_hex(16)

pred_test_img_paths_jpeg = glob('./static/uploads/test-set/images/*.jpeg')
pred_test_img_paths_jpg = glob('./static/uploads/test-set/images/*.jpg')
pred_test_img_paths = pred_test_img_paths_jpeg + pred_test_img_paths_jpg

demo_results_path = './static/uploads/demo_results/'
results_path = './static/uploads/results/'
upload_img_folder = './static/uploads/user-uploads/'




pred_img_paths_jpeg = glob('./static/uploads/user-uploads/*.jpeg')
pred_img_paths_jpg = glob('./static/uploads/user-uploads/*.jpg')
pred_img_paths = pred_img_paths_jpeg + pred_img_paths_jpg

demo_model = Yolov8(pred_test_img_paths, demo_results_path)
model = Yolov8(pred_img_paths, results_path)
 
@app.route('/')
def home():
     #return render_template('index.html', output_list=[], pred_img_paths=[], no_detect_imgs=[], sign_stamp_count=[], times={})
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_and_process_file():
    if request.method == 'POST':
        if 'file[]' not in request.files:
            flash('No file part')
            return redirect(request.url)

        files = request.files.getlist('file[]')

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                try:
                    file.save(os.path.join(upload_img_folder, filename))
                except Exception as e:
                    flash(f"Error saving file: {e}")

        flash('File(s) successfully uploaded')
        model.predict()
        model.save_output()

    return render_template('index.html')

        # Get user input value from front-end
        # input_invoice_ids = request.form['user-input-invoice-ids']
        # input_invoice_ids = input_invoice_ids.split(',')
        # input_ocr_model = request.form.get('input-ocr-model')
        
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
        #return render_template('index.html')

@app.route('/predict_demo_images')
def predict_demo_images():
    # Run the YOLOv8 prediction on the test images
    demo_model.predict()
    demo_model.save_output()

    # Get the image data for displaying in the table
    images_data = demo_model.get_image_data()
    # Print out the paths for debugging
    for original, processed, time, count in images_data:
        print(f"Original: {original}, Processed: {processed}, Time: {time}, Count: {count}")

    return render_template('demo_predict.html', images_data=images_data)

@app.route('/predict_images', methods=['GET', 'POST'])
def predict_images():
    
    model.predict()
    model.save_output()
    if request.method == 'POST':
        if 'file[]' not in request.files:
            flash('No file part')
            return redirect(request.url)

        files = request.files.getlist('file[]')

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(upload_img_folder, filename))

    model.predict()
    model.save_output()
    images_data = model.get_image_data()
    flash('File(s) successfully uploaded')
        
    # Get the image data for displaying in the table
    # Print out the paths for debugging
    for original, processed, time, count in images_data:
        print(f"Original: {original}, Processed: {processed}, Time: {time}, Count: {count}")

    return render_template('uploads.html', images_data=images_data)


@app.route('/display/demo-upload/<filename>')
def display_demo_upload_image(filename):
    return redirect(url_for('static', filename='uploads/test-set/images/' + filename), code=301)
    
@app.route('/display/demo-pred/<filename>')
def display_demo_pred_image(filename):
    return redirect(url_for('static', filename='uploads/demo_results/' + filename), code=301)

@app.route('/display/upload/<filename>')
def display_upload_image(filename):
    return redirect(url_for('static', filename='uploads/user-uploads/' + filename), code=301)
    
@app.route('/display/pred/<filename>')
def display_pred_image(filename):
    return redirect(url_for('static', filename='uploads/results/' + filename), code=301)

@app.route('/clear_data', methods=['POST'])
def clear_data():
    # Clear the user uploads and results folders
    clear_folder(upload_img_folder)
    clear_folder(results_path)

    # Reset the model's data

    flash('Data cleared successfully!')
    return redirect(url_for('home'))

def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8888)       