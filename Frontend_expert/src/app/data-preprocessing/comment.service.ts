import { Injectable } from '@angular/core';
import { HttpClient} from '@angular/common/http';

@Injectable({
  providedIn: 'root'
})
export class CommentService {
  private apiUrl = "http://127.0.0.1:5000/comments";

  constructor(private http: HttpClient) { }
 
 
      submitComment(data: any) {

      return this.http.post(this.apiUrl, data);
  }
}
