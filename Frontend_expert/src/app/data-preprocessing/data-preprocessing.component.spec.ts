import { ComponentFixture, TestBed } from '@angular/core/testing';

import { DataPreprocessingComponent } from './data-preprocessing.component';

describe('DataPreprocessingComponent', () => {
  let component: DataPreprocessingComponent;
  let fixture: ComponentFixture<DataPreprocessingComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ DataPreprocessingComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(DataPreprocessingComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
