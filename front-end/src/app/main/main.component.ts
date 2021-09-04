import { Component, Directive, OnInit } from '@angular/core';
// import { FileUploader } from 'ng2-file-upload';
import {
  HttpClient,
  HttpRequest,
  HttpEventType,
  HttpEvent
} from "@angular/common/http";
import { map, tap, last } from "rxjs/operators";
import { BehaviorSubject } from "rxjs";
import { FontAwesomeModule, FaIconLibrary  } from '@fortawesome/angular-fontawesome';
import { faCoffee, fas } from '@fortawesome/free-solid-svg-icons';
// import { faGoogle } from '@fortawesome/free-brands-svg-icons';
import { DomSanitizer } from '@angular/platform-browser';

const URL ="http://localhost:5000/upload_image";
//@Directive({ selector: '[ng2FileSelect]' })

@Component({
  selector: 'app-main',
  templateUrl: './main.component.html',
  styleUrls: ['./main.component.scss'],
  // class FileDropDirective
//Directive:[ng2FileSelect]
})


export class MainComponent implements OnInit {

  public progressSource = new BehaviorSubject<number>(0);
  // progress: number;
  infoMessage: any;
  isUploading: boolean = false;
  file!: File;
  isDivVisible :boolean = false;
  isstep2Processdone:boolean =false;

  step1ImgResponse : step1ResponseOBj ={
    img:"/assets/images/img1.jpg",
    car_name:"",
    file_name:"",
      class_id:"",
      img_ht: "",
      img_wt: "",
      bbox_coordinates:""
  };

  responseImg:ResponseObj = {
    img : "https://bulma.io/images/placeholders/480x480.png",
    car_name:'Not known'
  }

  imageUrl: string | ArrayBuffer =
    "https://bulma.io/images/placeholders/480x480.png";
  fileName: string = "No file selected";
  selectedgroup: any
  //modelList :{"Model1","Model2"}
  constructor(private http: HttpClient,private domSanitizer: DomSanitizer) {}

  ngOnInit() {
  console.log("main component")
  }


  onChange(file: File) {
    if (file) {
      this.fileName = file.name;
      this.file = file;

      const reader = new FileReader();
      reader.readAsDataURL(file);

      reader.onload = event => {
        this.imageUrl = reader.result;
      };
    }
  }

  onUpload() {
    this.infoMessage = null;
    this.isUploading = true;
    this.upload(this.file)
    this.upload(this.file).subscribe(message => {
      this.isUploading = true;
      this.infoMessage = message;
      console.log("message : "+message)
    });
  }


  upload(file: File) {
    let formData = new FormData();
    formData.append("File", file);
    console.log("selected model is "+ this.selectedgroup)
    formData.append("modelName",this.selectedgroup)

    const req = new HttpRequest(
      "POST",
      "http://localhost:3000/upload_image",
      formData,
      {
        reportProgress: true
      }
    );

    return this.http.request(req).pipe(
      map(event => this.getEventMessage(event, file),
      data => {
        console.log(data)
      }

      ),
      tap((envelope: any) => this.processProgress(envelope)),
      last()
    );

  }

  processProgress(envelope: any): void {
    if (typeof envelope === "number") {
      this.progressSource.next(envelope);
    }
  }

  tranform() {
    return this.domSanitizer.bypassSecurityTrustResourceUrl('data:image/png;base64,'+this.responseImg.img)
  }

  step1transform() {
    return this.domSanitizer.bypassSecurityTrustResourceUrl('data:image/png;base64,'+this.step1ImgResponse.img)
  }
  onReset() {
    this.responseImg.img = "";
   // this.file= new File(["blank"],"blank.txt");
    this.responseImg.car_name="Not Known"
    console.log("reset done")
  }
  private getEventMessage(event: HttpEvent<any>, file: File) {
    switch (event.type) {
      case HttpEventType.Sent:
        return `Uploading file "${file.name}" of size ${file.size}.`;
      case HttpEventType.UploadProgress:
        return Math.round((100 * event.loaded) );
      case HttpEventType.Response:
        console.log(event)

        this.responseImg= event.body;
        this.isstep2Processdone = true;
        return `File "${file.name}" was completely uploaded!`;
      default:
        return `File "${file.name}" surprising upload event: ${event.type}.`;
    }
  }
  private getStep1EventMessage(event: HttpEvent<any>, img) {
    switch (event.type) {
      case HttpEventType.Sent:
        return "Uploading image" ;
      case HttpEventType.UploadProgress:
        return Math.round((100 * event.loaded) / event.total);
      case HttpEventType.Response:
        console.log(event)

        this.step1ImgResponse= event.body;
        this.isDivVisible =true;
        console.log("image response received")
        return `image uploaded was completely uploaded!`;
      default:
        return `image surprising upload event: ${event.type}.`;
    }
  }
  tagme(event) {
    console.log(event.target.name)
    var imageName = event.target.name
    let formData = new FormData();
    formData.append("imageName", imageName);

    this.http.post<any>("http://localhost:3000/step1", formData).pipe(
      map(res=> {
        console.log("from pipe")
        this.step1ImgResponse.img=res["img"];
        this.step1ImgResponse=res;

        this.isDivVisible=true;
        console.log(res);
      })


    ).subscribe(
      data =>{
      (res) => this.step1ImgResponse.img;
      (err) => console.log(err);
      ()=>console.log("completed...")
      }
    );
  }

}

export class ResponseObj {
  img: string | ArrayBuffer ;
  car_name:String
}

export class step1ResponseOBj {
  img:string|ArrayBuffer;
  car_name:string;
  file_name:string;
  class_id:string;
  img_ht:string;
  img_wt:string;
  bbox_coordinates:string;
}
