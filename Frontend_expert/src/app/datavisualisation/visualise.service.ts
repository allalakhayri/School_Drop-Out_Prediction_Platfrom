import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Injectable({
  providedIn: 'root'
})
export class VisualiseService {

private apiUrl = 'http://localhost:5000/visualise'; 

constructor(private http: HttpClient) { }

visualiseData(data :any ) {
  return this.http.post(this.apiUrl, data);
}
}
