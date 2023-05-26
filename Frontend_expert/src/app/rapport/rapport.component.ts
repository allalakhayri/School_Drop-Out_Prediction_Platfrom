import { Component, OnInit } from '@angular/core';
import { GenerateService} from './generate.service'
import { ResponseDataService } from '../response-data.service'
import * as fileSaver from 'file-saver';


@Component({
  selector: 'app-rapport',
  templateUrl: './rapport.component.html',
  styleUrls: ['./rapport.component.css']
})

export class RapportComponent implements OnInit{
  dataGenerated = false;
  error: string | null = null;
  time_tot: number | null = null; 
  time_preprocessing: number | null = null; 
  time_processing: number | null = null; 
  Important_features :any = null;
  date: Date = new Date();
  cat_cols: string[] | null = null;
  num_cols: string[] | null = null;
  null_counts!: number; 
  target !:string ; 
  classif_report_rf !:string ; 
  classif_report_clf !:string ; 
  classif_report_gbm !:string ; 
  classif_report_dt !:string ; 
  classif_report_svm !:string ; 
  

  constructor(private responseJSON: ResponseDataService,
    private generateService: GenerateService,
    ){}

    ngOnInit(): void {  
      const data = this.responseJSON.getResponseData();
      console.log(data);
      this.generateService.generateData(data).subscribe(
        (result: any) => {
          this.dataGenerated = true;
          console.log(result);
          this.time_tot=result.Time_tot ; 
          this.cat_cols=result.Categorical_cols; 
          this.num_cols=result.Numerical_cols;
          this.Important_features=result.Important_features;
          this.time_preprocessing=result.Time_preprocessing; 
          this.time_processing=result.Time_processing; 
          this.classif_report_clf=result.classif_report_clf;
          this.classif_report_rf=result.classif_report_rf;
          this.classif_report_dt=result.classif_report_dt;
          this.classif_report_gbm=result.classif_report_gbm;
          this.classif_report_svm=result.classif_report_svm;
          console.log(this.time_tot)
          console.log(this.time_preprocessing)
          console.log(this.time_processing)
          console.log(this.cat_cols)
          console.log(this.num_cols)
          console.log(this.Important_features)
          console.log(this.classif_report_rf);
          
},
(error: any) => { 
  console.log("error while generating report! ");
} );
    }
    download() {
      const pdfUrl = '../files/report.pdf';
      fileSaver.saveAs(pdfUrl, 'report.pdf');
    }

  }