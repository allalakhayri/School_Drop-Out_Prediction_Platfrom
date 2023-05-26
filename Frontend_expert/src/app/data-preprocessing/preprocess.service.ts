import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Injectable({
  providedIn: 'root'
})
export class PreprocessService {

  private apiUrl = 'http://localhost:5000/preprocess'; 

  constructor(private http: HttpClient) { }

  preprocessData(data :any ) {
    return this.http.post(this.apiUrl, data);
  }
}
