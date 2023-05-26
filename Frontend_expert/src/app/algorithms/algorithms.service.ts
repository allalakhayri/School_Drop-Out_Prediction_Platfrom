import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root'
})
export class AlgorithmsService {
 
  private apiUrl = 'http://127.0.0.1:5000/algorithms'; 

  constructor(private http: HttpClient) { }

  public generateAlgo(data :any ) {
    return this.http.post(this.apiUrl, data);
  }
}