from flask import Flask,render_template,request,redirect,url_for,send_from_directory
from constants.items import *
import os
from werkzeug.utils import secure_filename
from util.db import read_db,write_project,find_project_by_id,find_project_by_id_and_delete, about_db, write_about
from util.app_project import create_project, get_projects, get_projects_by_id, update_project_by_id, delete_project_by_id
app = Flask(__name__)

app.config["UPLOAD_FOLDER"]="files"

@app.route("/")
def index():
    About = about_db()
    return render_template('/views/home.html',APP_NAME=APP_NAME,MENU_ITEMS=MENU_ITEMS,SOCIAL_LINKS=SOCIAL_LINKS,MY_PROJECTS=MY_PROJECTS,About=About)
 
@app.route("/dashboard")
def dashboard():
    return  render_template("/views/dashboard/index.html",APP_NAME=APP_NAME,DASHBOARD_MENU=DASHBOARD_MENU)


@app.route("/dashboard/files",methods=["GET","POST"])
def files():
    if request.method=="POST":
        file = request.files["file"]
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config["UPLOAD_FOLDER"],filename))
        return redirect(url_for("files"))
       # FİLE UPLOAD
    else:
        files = os.listdir(os.path.join(app.config["UPLOAD_FOLDER"]))
        return render_template("/views/files/index.html",APP_NAME=APP_NAME,DASHBOARD_MENU=DASHBOARD_MENU,files=files)



@app.route("/dashboard/projects")
def projects():
    all_projects = get_projects()
    # print(all_projects)
    # for prj in all_projects:
    #     for key in prj:
    #         print(prj[key])
    return render_template("/views/projects/index.html",APP_NAME=APP_NAME,DASHBOARD_MENU=DASHBOARD_MENU,all_projects=all_projects)


@app.route("/dashboard/projects/<string:id>/delete",methods=["GET","POST"])
def project_delete(id):
    if request.method=="POST":
        print(id)
        delete_project_by_id(id)
        return redirect(url_for("projects"))
    else:
        # project = get_projects_by_id(id)
        # print(project)
        return redirect(url_for("projects"))


@app.route("/dashboard/projects/modify/<string:id>",methods=["GET","POST"])
def project_modify(id):
    if request.method=="POST":
        print('\n\n\n', id, '\n\n\n')
        #  grab values from form and write into csv file
        title = request.form.get("title")
        description = request.form.get("description")
        cover = request.form.get("cover")
        githubLink = request.form.get("githubLink")
        liveLink = request.form.get("liveLink")


        update = {
            'title': title,
            'description': description,
            'cover': cover,
            'github_link': githubLink,
            'live_link': liveLink
        }


        # write_project(title,description,cover,githubLink,liveLink)
        update_project_by_id(id, update)
        return redirect(url_for("projects"))
    else:
        prj = get_projects_by_id(id)
        print('\n\n\n', prj, '\n\n\n')
    return render_template("/views/projects/modify.html",APP_NAME=APP_NAME,DASHBOARD_MENU=DASHBOARD_MENU, project=prj)

# @app.route("/dashboard/projects/<string:id>",methods=["GET","POST"])
# def project_modify(id, update):
#     if request.method=="POST":
#         print(id)
#         update_project_by_id(id, update)
#         return redirect(url_for("projects"))
#     else:
#         project = get_projects_by_id(id)
#         print(project)
#         return redirect(url_for("projects"))

# @app.route("/dashboard/projects/modify/<string:id>",methods=["GET","POST"])
# def modify_project(id):
#     if request.method=="POST":
#         # grab values from form and write into csv file
#         title = request.form.get("title")
#         description = request.form.get("description")
#         cover = request.form.get("cover")
#         githubLink = request.form.get("githubLink")
#         liveLink = request.form.get("liveLink")
#         update = {
#             'title': title,
#             'description': description,
#             'cover': cover,
#             'github_link': githubLink,
#             'live_link': liveLink
#         }
#         # write_project(title,description,cover,githubLink,liveLink)
#         update_project_by_id(id, update)
#         return redirect(url_for("projects"))
#     else:
#         # display form for adding project
#         project = get_projects_by_id(id)
#         return render_template("/views/projects/modify.html",APP_NAME=APP_NAME,DASHBOARD_MENU=DASHBOARD_MENU, project=project)


@app.route("/dashboard/projects/new",methods=["GET","POST"])
def new_project():
    if request.method=="POST":
        # grab values from form and write into csv file
        title = request.form.get("title")
        description = request.form.get("description")
        cover = request.form.get("cover")
        githubLink = request.form.get("githubLink")
        liveLink = request.form.get("liveLink")
        # write_project(title,description,cover,githubLink,liveLink)
        create_project(title,description,cover,githubLink,liveLink)
        return redirect(url_for("projects"))
    else:
        # display form for adding project
        return render_template("/views/projects/new.html",APP_NAME=APP_NAME,DASHBOARD_MENU=DASHBOARD_MENU)


@app.route("/dashboard/files/<string:filename>",methods=["GET","POST"])
def file_actions(filename):
    file_path = os.path.join(app.config["UPLOAD_FOLDER"],filename)
    if request.method=="POST":
        os.remove(file_path)
        return redirect(url_for("files"))
    else:
        return send_from_directory(path=app.root_path,directory=app.config["UPLOAD_FOLDER"],filename=filename)

@app.route("/dashboard/about", methods=["GET", "POST"])
def about():
    return render_template("/views/about/index.html",APP_NAME=APP_NAME,DASHBOARD_MENU=DASHBOARD_MENU)

@app.route("/dashboard/modify_about", methods=["GET", "POST"])
def get_about():
    if request.method == "GET":
        about = about_db()
        return render_template("/views/about/index.html",APP_NAME=APP_NAME,DASHBOARD_MENU=DASHBOARD_MENU,About=about)
    else:
        text = request.form.get("About")
        write_about(text)
        return redirect(url_for("about"))


# @app.route("/dashboard/save_about", methods=["GET", "POST"])
# def save_about():
#     if request.method == "POST":
#         text = request.form.get("About")
#         write_about(text)
#         return redirect(url_for("about"))




if __name__ == '__main__':
    app.run(debug=True)
