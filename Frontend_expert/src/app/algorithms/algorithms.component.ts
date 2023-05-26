import { Component, Input, OnInit,AfterViewInit } from '@angular/core';
import { AlgorithmsService } from './algorithms.service';
import { ActivatedRoute, Router } from '@angular/router';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { FileUploadService } from '../file-upload/file-upload.service';
import {Location, getLocaleCurrencyCode} from '@angular/common'
import { HttpClient } from '@angular/common/http';
import {ResponseDataService} from '../response-data.service'
@Component({
  selector: 'app-algorithms',
  templateUrl: './algorithms.component.html',
  styleUrls: ['./algorithms.component.css']
})
export class AlgorithmsComponent implements OnInit {
  responseJSON!:any ;
  algoGenerated= true;
  accuracy!:any; 
  error: string | null = null;
  comment!:string; 
  cm:string='../../assets/noimg.png'
  curve:string='../../assets/noimg.png'
  acc:string='../../assets/noimg.png'
  report:string='../../assets/noimg.png'
 formDat!:FormGroup
  constructor(private fb:FormBuilder,
    private responseService: ResponseDataService,
    private algoService: AlgorithmsService,
    private router: Router,private fileUploadService:FileUploadService,private location:Location,private route:ActivatedRoute,private http: HttpClient,private algoServices:AlgorithmsService
    ){}
ngOnInit(): void {
  this.responseJSON = this.responseService.getResponseData();
  const data = this.responseJSON;
  console.log(this.responseJSON);

  this.algoService.generateAlgo(data).subscribe(
    (result: any) => {
      this.algoGenerated=true; 
      console.log(result);
      
    }, 
    (error: any) => {
      console.log(error);}
  );
  this.formDat=this.fb.group({

    name:'',
    comment:['',Validators.required],
    })
}

executeLR() {
 
    this.cm='../../assets/LR_Cm.png';
    this.curve='../../assets/LRCurve.png';
    this.acc='../../assets/accuracyLR.png'

    this.report='../../assets/c_reportLR.png'

}
executeRF() {


    this.cm='../../assets/RF_Cm.png';
    this.curve='../../assets/RFCurve.png';
    this.acc='../../assets/accuracyRF.png'
  
    this.report='../../assets/c_reportRF.png'

}
executeDT() {
    this.cm='../../assets/DT_Cm.png';
    this.curve='../../assets/DTCurve.png';
    this.acc='../../assets/accuracyDT.png'

    this.report='../../assets/c_reportDT.png'
}
executeGBC() {
    this.cm='../../assets/GBM_Cm.png';
    console.log(this.cm);
    this.curve='../../assets/GBMCurve.png';
    this.acc='../../assets/accuracyGBM.png'
    this.report='../../assets/c_reportGBM.png'


}
executeSVM() {
    this.cm='../../assets/SVM_Cm.png';
    this.curve='../../assets/SVMCurve.png';
    this.acc='../../assets/accuracySVM.png'
    this.report='../../assets/c_reportSVM.png'
 
}
retry() {
  this.error = null;
  window.location.reload();
  this.ngOnInit();
}

addComment(a:any){
  this.formDat.value.name=a
  //this.algoService.addComment(this.formDat.value).subscribe({
    //next:(res=>{
     // if(res.Message=="Success !"){
      //this.toast.success({detail:"SUCCESS",summary:res.Message,duration:5000});
    //this.formDat.reset()}
   // }),
   // error:(err=>{
      // this.toast.error({detail:"ERROR",summary:"Something went wrong !",duration:5000});
    //})
 // })
//}

}
}