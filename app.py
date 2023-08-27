from flask import Flask, render_template, request
import analysis

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/launch')
def launch():
    return render_template('launch.html')


@app.route('/submit', methods=['POST'])
def submit():
    text = request.form['text']
    show_loading_overlay = True
    result = analysis.get_compellingness_ai(text)
    descriptions = analysis.get_compellingness_description(result)
    warnings = analysis.ethics_warning(text)
    pathos = analysis.get_pathos_description(result)
    logos = analysis.get_logos_description(result)
    return render_template('results.html', text=text, result=result, descriptions=descriptions, warnings=warnings, pathos=pathos, logos=logos, show_loading_overlay=show_loading_overlay)


if __name__ == '__main__':
    app.run()
