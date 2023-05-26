import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root'
})
export class ProcessService {

private apiUrl = 'http://localhost:5000/process'; 

constructor(private http: HttpClient) { }

processData(data :any ) {
  return this.http.post(this.apiUrl, data);
}
}
