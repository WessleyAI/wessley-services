import { Module } from '@nestjs/common';
import { SupabaseService } from './supabase.service';
import { NotificationService } from './notification.service';

@Module({
  providers: [
    SupabaseService,
    NotificationService
  ],
  exports: [
    SupabaseService,
    NotificationService
  ]
})
export class RealtimeModule {}