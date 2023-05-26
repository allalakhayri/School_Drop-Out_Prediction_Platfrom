import { Component, ElementRef, OnInit, ViewChild } from '@angular/core';
import { FormGroup, FormControl, Validators, FormBuilder } from '@angular/forms';
import {FileUploadService} from './file-upload.service'
import { Router, NavigationExtras } from '@angular/router';
import {ResponseDataService} from '../response-data.service'

@Component({
  selector: 'app-upload-file',
  templateUrl: './file-upload.component.html',
  styleUrls: ['./file-upload.component.css']
})
export class UploadFileComponent implements OnInit {
  [x: string]: any;
  uploadForm!: FormGroup;
  selectedFile!: File;
  csvFile!: File;
  @ViewChild('fileInput') fileInput!: ElementRef;
  responseMessage: string | null = null;
  errorMessage: string | null = null;
  responseJSON: any;

  constructor(private fb:FormBuilder,
    private fileUploadService: FileUploadService,
    private router: Router,
    private elementRef: ElementRef,
    private responseDataService: ResponseDataService,
    )
    {}
   
   fileUploaded = false;
ngOnInit(){
  this.fileUploaded = false;
} 
  fileToUpload: File | null = null;
  onFileSelected(event: any) {
    this.fileToUpload = event.target.files[0];
  }
  onUpload() {
  
    if (!this.fileToUpload) {
      this.errorMessage = "Please select a file to upload.";
      return;
    }
    if (this.fileToUpload.type !== 'text/csv') {
      this.errorMessage = "Invalid file type ! Only CSV files are allowed.";
      return;
    }
    const formData = new FormData();
  if (this.fileToUpload) {
    formData.append('file', this.fileToUpload, this.fileToUpload.name);
    
  } else {
    console.log("No selected file !");
    
  }
    this.fileUploadService.uploadFile(formData).subscribe(
      (response) => {
       this.fileUploaded = true;
       this.elementRef.nativeElement.querySelector('#response-message').innerHTML =
       "File uploaded succesfully !"
       let csvData=response['data'];
       const rows = csvData.split('\n');
       let keys = rows[0].split(',');
       // Remove '\r' from last key
    keys[keys.length - 1] = keys[keys.length - 1].replace(/\r/g, '');
      
       const jsonData = []; 

     for (let i = 1; i < rows.length-1; i++) {  
      const row = rows[i];
      const values = row.split(',');
   // Remove '\r' from each value
    for (let j = 0; j < values.length; j++) {
     values[j] = values[j].replace(/\r/g, '');
}
     
       const obj: { [key: string]: any } = {};

       for (let j = 1; j < keys.length; j++) {
        let value = values[j];
          obj[keys[j]] = value;
          
 }
    console.log(obj);
   jsonData.push(obj); 
   
   }
 this.responseDataService.setResponseData(jsonData);



    },
      (error) => { console.log(error)
      this.errorMessage = error.message;
      this.fileUploaded = false;
    }
    );
    ;
  }
}