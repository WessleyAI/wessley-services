import { 
  Injectable, 
  CanActivate, 
  ExecutionContext, 
  UnauthorizedException,
  Logger 
} from '@nestjs/common';
import { Reflector } from '@nestjs/core';
import { Request } from 'express';
import { SupabaseService } from '../modules/realtime/supabase.service';

export interface AuthenticatedRequest extends Request {
  user?: {
    id: string;
    email: string;
    role?: string;
    aud?: string;
  };
}

@Injectable()
export class AuthGuard implements CanActivate {
  private readonly logger = new Logger(AuthGuard.name);

  constructor(
    private readonly supabaseService: SupabaseService,
    private readonly reflector: Reflector
  ) {}

  async canActivate(context: ExecutionContext): Promise<boolean> {
    // Check if the route is marked as public
    const isPublic = this.reflector.getAllAndOverride<boolean>('isPublic', [
      context.getHandler(),
      context.getClass(),
    ]);

    if (isPublic) {
      return true;
    }

    // Check if Supabase is configured
    if (!this.supabaseService.isConfigured()) {
      this.logger.warn('Supabase not configured - allowing request without authentication');
      return true; // Allow requests when Supabase is not configured (development mode)
    }

    const request = context.switchToHttp().getRequest<AuthenticatedRequest>();
    const token = this.extractTokenFromHeader(request);

    if (!token) {
      throw new UnauthorizedException('Authorization token required');
    }

    try {
      const user = await this.supabaseService.verifyAndGetUser(token);
      
      if (!user) {
        throw new UnauthorizedException('Invalid or expired token');
      }

      // Attach user to request
      request.user = {
        id: user.id,
        email: user.email || '',
        role: user.role || 'user',
        aud: user.aud
      };

      this.logger.debug(`Authenticated user: ${user.email} (${user.id})`);
      return true;

    } catch (error) {
      this.logger.warn(`Authentication failed: ${error.message}`);
      throw new UnauthorizedException('Authentication failed');
    }
  }

  private extractTokenFromHeader(request: Request): string | undefined {
    const authHeader = request.headers.authorization;
    
    if (!authHeader) {
      return undefined;
    }

    // Support both "Bearer <token>" and "<token>" formats
    if (authHeader.startsWith('Bearer ')) {
      return authHeader.substring(7);
    }

    return authHeader;
  }
}

// Decorator to mark routes as public (no authentication required)
import { SetMetadata } from '@nestjs/common';

export const Public = () => SetMetadata('isPublic', true);