import {  Component, ElementRef, OnInit, ChangeDetectorRef  } from '@angular/core';
import { ResponseDataService } from '../response-data.service'
import { CommentService } from '../data-preprocessing/comment.service'
import { ProcessService} from './process.service'

@Component({
  selector: 'app-data-processing',
  templateUrl: './data-processing.component.html',
  styleUrls: ['./data-processing.component.css']
})

export class DataProcessingComponent implements OnInit{
  dataProcessed = false;
  comment: string = '';
  error: string | null = null;
  feature_importances :any = null;
  time_taken: number | null = null; 
  constructor(private responseJSON: ResponseDataService,
    private cdRef: ChangeDetectorRef,
    private commentService: CommentService,
    private processService: ProcessService,
    ) {}
  ngOnInit(): void {  
    const data = this.responseJSON.getResponseData();
    console.log(data);
    this.processService.processData(data).subscribe(
      (result: any) => {
        this.dataProcessed = true;
        console.log(data);
        this.feature_importances = result.feature_importances;
        console.log(this.feature_importances);
        this.time_taken=result.time_taken;
        this.cdRef.detectChanges();

  },
  (error: any) => { 
    console.log("error while importing plots! ");
  } );
  }

  submitForm() {
    const formData = {
      comment: this.comment
    };
    console.log(formData);
    if (this.error) {
      this.dataProcessed = false;
  
    }
    this.commentService.submitComment(formData).subscribe(
      (result: any) => {
        this.dataProcessed = true;
  
  
        console.log(result);
      },
      (error: any) => {
        this.error = error.message;
        this.dataProcessed = false;
  
        console.log(error);
      }
    );
  }
  retry() {
    this.error = null;
    window.location.reload();
    this.ngOnInit();
  }
  

}
