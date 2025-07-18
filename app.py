from flask import Flask, render_template, Response,jsonify,request,session,send_file

#FlaskForm--> it is required to receive input from the user
# Whether uploading a video file  to our object detection model

from flask_wtf import FlaskForm


from wtforms import FileField, SubmitField,StringField,DecimalRangeField,IntegerRangeField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired,NumberRange
import os
import cv2


# Required to run the YOLOv8 model
import cv2

# YOLO_Video is the python file which contains the code for our object detection model
#Video Detection is the Function which performs Object Detection on Input Video
from YOLO_Video import video_detection
from image_detection import image_detection
app = Flask(__name__)

app.config['SECRET_KEY'] = 'vinayak'
app.config['UPLOAD_FOLDER'] = 'static/files'


#Use FlaskForm to get input video file  from user
class UploadFileForm(FlaskForm):
    #We store the uploaded video file path in the FileField in the variable file
    #We have added validators to make sure the user inputs the video in the valid format  and user does upload the
    #video when prompted to do so
    file = FileField("File",validators=[InputRequired()])
    submit = SubmitField("Run")


def generate_frames(path_x = ''):
    try:
        if not path_x or path_x == '':
            # Use a default video if no path provided
            path_x = 'static/files/ppe-1.mp4'
        
        yolo_output = video_detection(path_x)
        if yolo_output is None:
            return
            
        for detection_ in yolo_output:
            ref,buffer=cv2.imencode('.jpg',detection_)
            frame=buffer.tobytes()
            yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')
    except Exception as e:
        print(f"Error in generate_frames: {e}")
        return

def generate_frames_web(path_x):
    try:
        yolo_output = video_detection(path_x)
        if yolo_output is None:
            return
            
        for detection_ in yolo_output:
            ref,buffer=cv2.imencode('.jpg',detection_)
            frame=buffer.tobytes()
            yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')
    except Exception as e:
        print(f"Error in generate_frames_web: {e}")
        return

@app.route('/', methods=['GET','POST'])
@app.route('/home', methods=['GET','POST'])
def home():
    session.clear()
    return render_template('indexproject.html')

@app.route('/image_upload', methods=['GET'])
def image_upload_page():
    return render_template('image_upload.html')
# Rendering the Webcam Rage
#Now lets make a Webcam page for the application
#Use 'app.route()' method, to render the Webcam page at "/webcam"
@app.route("/webcam", methods=['GET','POST'])

def webcam():
    session.clear()
    return render_template('ui.html')
@app.route('/FrontPage', methods=['GET','POST'])
def front():
    # Upload File Form: Create an instance for the Upload File Form
    form = UploadFileForm()
    if form.validate_on_submit():
        # Our uploaded video file path is saved here
        file = form.file.data
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                               secure_filename(file.filename)))  # Then save the file
        # Use session storage to save video file path
        session['video_path'] = os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                                             secure_filename(file.filename))
    return render_template('videoprojectnew.html', form=form)
@app.route('/video')
def video():
    #return Response(generate_frames(path_x='static/files/bikes.mp4'), mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(generate_frames(path_x = session.get('video_path', None)),mimetype='multipart/x-mixed-replace; boundary=frame')

# To display the Output Video on Webcam page
@app.route('/webapp')
def webapp():
    #return Response(generate_frames(path_x = session.get('video_path', None),conf_=round(float(session.get('conf_', None))/100,2)),mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(generate_frames_web(path_x=0), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect_image', methods=['POST'])
def detect_image():
    """Detect PPE in uploaded image and return the result"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file:
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Run detection
            result_img = image_detection(filepath)
            
            if result_img is not None:
                # Save result image
                result_filename = f"detected_{filename}"
                result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
                cv2.imwrite(result_path, result_img)
                
                return jsonify({
                    'success': True,
                    'original_image': f"/static/files/{filename}",
                    'detected_image': f"/static/files/{result_filename}"
                })
            else:
                return jsonify({'error': 'Detection failed'}), 500
                
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    import os
    
    port = int(os.environ.get("PORT", 5000))  # fallback for local dev
    app.run(host='0.0.0.0', port=port)