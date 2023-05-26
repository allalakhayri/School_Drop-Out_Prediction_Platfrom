
import { ComponentFixture, TestBed } from '@angular/core/testing';

import { UploadFileComponent } from './file-upload.component';

describe('FileUploadComponent', () => {
  let component: UploadFileComponent ;
  let fixture: ComponentFixture<UploadFileComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ UploadFileComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent( UploadFileComponent );
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});