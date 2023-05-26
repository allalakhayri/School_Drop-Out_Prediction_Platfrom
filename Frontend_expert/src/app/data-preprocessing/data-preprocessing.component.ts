import { Component ,OnInit ,ChangeDetectorRef  } from '@angular/core';
import {ResponseDataService} from '../response-data.service'
import {PreprocessService} from './preprocess.service'
import { CommentService } from './comment.service'
 @Component({
  selector: 'app-data-preprocessing',
  templateUrl: './data-preprocessing.component.html',
  styleUrls: ['./data-preprocessing.component.css']
})

  export class DataPreprocessingComponent implements OnInit{
    responseJSON!:any ;
    formatted_stats: any=null;
    cat_cols: string[] | null = null;
    num_cols: string[] | null = null;
    time_taken: number | null = null;
    error: string | null = null;
    comment: string = '';
    outliers :any=null; 
    images: string[] = [];
    dataPreprocessed = false;
    constructor(private responseService: ResponseDataService ,

      private cdRef: ChangeDetectorRef,
      private preprocessService: PreprocessService,
      private commentService: CommentService,
      private responseDataService: ResponseDataService) { }
    ngOnInit(): void {
      this.dataPreprocessed = false;
        this.responseJSON = this.responseService.getResponseData();
        console.log(this.responseJSON);
       
        const data = this.responseJSON;
        this.preprocessService.preprocessData(data).subscribe(
          (result: any) => {
            console.log(result);
            this.responseJSON = result.data;
            this.formatted_stats = result.formatted_stats;
            this.cat_cols = result.cat_cols;
            this.num_cols = result.num_cols;
            this.time_taken = result.time_taken;
            this.outliers=result.outliers; 
            this.images=result.plots;
            this.dataPreprocessed = true; 
            console.log(this.responseJSON);
            console.log(this.formatted_stats);
            console.log(this.cat_cols);
            console.log(this.num_cols);
            console.log(this.time_taken);
            console.log(this.images);
            console.log( this.outliers);
            console.log( this.images);
            this.cdRef.detectChanges();
          },
          (error: any) => {
            console.log(error);
            if (error.status === 400) {
              this.responseJSON = null; 
              this.error= 'Error while preprocessing data , try again !' + error.error.message;
            } else if (error.status === 404) {
              this.responseJSON = null;  
              this.error= 'Error while preprocessing data , try again ! ' + error.error.message;
            } else if (error.status === 500) {
              this.responseJSON = null;
              this.error = 'Error while preprocessing data , try again !  ' + error.error.message;
            } else {
              this.responseJSON = null;   
              this.error = 'The data you uploaded is not compatible ! ';
            }
          }
        );

      }

  
    submitForm() {
      const formData = {
        formatted_stats: this.formatted_stats,
        cat_cols: this.cat_cols,
        num_cols: this.num_cols,
        time_taken: this.time_taken,
        comment: this.comment
      };
      console.log(formData);
      if (this.error) {
        this.dataPreprocessed = false;

      }
      this.commentService.submitComment(formData).subscribe(
        (result: any) => {
          this.dataPreprocessed = true;
          console.log(result);
        },
        (error: any) => {
          this.error = error.message;
          this.dataPreprocessed = false;
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
