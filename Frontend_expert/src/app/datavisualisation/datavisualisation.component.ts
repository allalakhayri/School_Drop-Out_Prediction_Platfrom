import {  Component, ElementRef, OnInit, ViewChild } from '@angular/core';
import { ResponseDataService } from '../response-data.service'
import { VisualiseService} from './visualise.service'
import { CommentService } from '../data-preprocessing/comment.service'
@Component({
  selector: 'app-datavisualisation',
  templateUrl: 'datavisualisation.component.html',
  styleUrls: ['./datavisualisation.component.css']
})
export class DatavisualisationComponent implements OnInit {
  dataVisualised = false;
  images: string[] = [];
  time_taken: any | null = null;
  comment: string = '';
  error: string | null = null;

constructor(private responseJSON: ResponseDataService,
  private visualiseService: VisualiseService,
  private commentService: CommentService,) { }
ngOnInit(): void {  
  const data = this.responseJSON.getResponseData();
  console.log(data);
  this.visualiseService.visualiseData(data).subscribe(
    (result: any) => {
      this.dataVisualised = true;
      console.log(result);
      this.images=result.data ;
      this.time_taken=result.time_taken;
      console.log(this.images);
      console.log(this.time_taken);
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
    this.dataVisualised = false;

  }
  this.commentService.submitComment(formData).subscribe(
    (result: any) => {
      this.dataVisualised = true;
      console.log(result);
    },
    (error: any) => {
      this.error = error.message;
      this.dataVisualised = false;

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